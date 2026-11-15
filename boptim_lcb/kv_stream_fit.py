"""A second stand-in for the cache entries of a token that left after loop 1: derive them from the token's FINAL loop-1 state.

The copy rule fills the loop-2..4 slot of layer l with the loop-1 key/value of layer l, which was computed from the residual
stream as it stood inside loop 1 at layer l (at layer 0 that is the bare token embedding). What loops 2..4 would really see
at layer l is the stream as it stands inside THEIR pass, which starts from the finished loop-1 state. The "stream" stand-in is
  key_l, value_l = k_proj_l, v_proj_l ( input_layernorm_l ( finished loop-1 state ) )
i.e. each layer's own projections applied to the state the next loop starts from: exact at layer 0, drifting with depth.
Cost: one normalisation and two projections per layer, about 15 % of one loop, once per exited token.

Reports, by layer, the relative error (against a constant, held-out tokens) of keys and values for loop 1 -> 2, 3, 4:
  copy | affine(copy) | stream | affine(stream) | affine(stream and copy together)
and saves the affine(stream and copy) maps when a path is given.

  CUDA_VISIBLE_DEVICES=0 python boptim_lcb/kv_stream_fit.py [n_fit] [n_test] [--save path]
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = "/srv/scratch/jli199/bfcl_stage/bfcl_search.jsonl"
WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
TARGETS = [1, 2, 3]          # loops 2, 3, 4 (zero-based) for a token that left after loop 1
RIDGE = 1e-2


def fit(model, tok, n_fit=100, n_test=32, save=None, log=print):
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    full_dir = None
    for rec in sorted(glob.glob(f"{WORK}/q1.000-1.000-1.000_*/record.json")):
        if json.load(open(rec)).get("status") == "ok":
            full_dir = rec.rsplit("/", 1)[0]
    gens = [json.loads(l) for l in open(f"{full_dir}/gen.jsonl") if l.strip() and '"summary"' not in l[:12]]
    cand = [i for i in range(len(rows)) if i % 4 != 0]            # never the items of the teacher-forced exit test
    step = max(1, len(cand) // (n_fit + n_test))
    pick = cand[::step][: n_fit + n_test]
    fit_items, test_items = [], []
    for j, i in enumerate(pick):
        (test_items if j % 4 == 3 and len(test_items) < n_test else fit_items).append(i)
    inner = model.model
    NL = model.config.num_hidden_layers

    def features(i):
        """per kind (keys, values): true [3 targets, layers, heads, n, D], copy [layers, heads, n, D], stream [layers, heads, n, D]"""
        r, g = rows[i], gens[i]
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"].rstrip()}], tokenize=False, add_generation_prompt=True) + "["
        p_ids = tok(text)["input_ids"]
        a_ids = tok(g["gen"][1:] if g["gen"].startswith("[") else g["gen"], add_special_tokens=False)["input_ids"]
        if not a_ids:
            return None
        ids = torch.tensor([p_ids + a_ids], device="cuda")
        out, hidden_list, _ = inner(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        pk = out.past_key_values
        lo = len(p_ids)
        cos, sin = inner.rotary_emb(torch.zeros(1, device="cuda", dtype=torch.float32), torch.arange(ids.shape[1], device="cuda")[None])
        cos, sin = cos[0, lo:].float(), sin[0, lo:].float()

        def unrot(k):
            k = k.float()
            half = k.shape[-1] // 2
            return k * cos - torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin

        h = hidden_list[0][0, lo:]                                  # finished loop-1 state of the answer tokens, [n, hidden]
        n = h.shape[0]
        res = []
        for kind in (0, 1):
            cache = pk.key_cache if kind == 0 else pk.value_cache
            get = (lambda s, l: unrot(cache[s * NL + l][0, :, lo:, :])) if kind == 0 else (lambda s, l: cache[s * NL + l][0, :, lo:, :].float())
            true = torch.stack([torch.stack([get(s, l) for l in range(NL)]) for s in TARGETS]).double().cpu()
            copy = torch.stack([get(0, l) for l in range(NL)]).double().cpu()
            st = []
            for l in range(NL):
                layer = inner.layers[l]
                x = layer.input_layernorm(h)
                proj = layer.self_attn.k_proj if kind == 0 else layer.self_attn.v_proj
                st.append(proj(x).view(n, -1, true.shape[-1]).transpose(0, 1).float())
            res.append((true, copy, torch.stack(st).double().cpu()))
        del pk, out, hidden_list
        return res

    def design(copy, stream, mode):
        ones = torch.ones(*copy.shape[:3], 1, dtype=torch.float64)
        parts = {"copy": [copy], "stream": [stream], "both": [stream, copy]}[mode]
        return torch.cat(parts + [ones], dim=-1)

    modes = ["copy", "stream", "both"]
    acc = {}
    with torch.inference_mode():
        for nn_, i in enumerate(fit_items):
            f = features(i)
            if f is None:
                continue
            for kind, (true, copy, stream) in enumerate(f):
                for mode in modes:
                    X = design(copy, stream, mode)
                    xtx = X.transpose(-1, -2) @ X
                    for t in range(len(TARGETS)):
                        key = (kind, mode, t)
                        if key not in acc:
                            acc[key] = [torch.zeros_like(xtx), torch.zeros(*xtx.shape[:2], X.shape[-1], true.shape[-1], dtype=torch.float64)]
                        acc[key][0] += xtx
                        acc[key][1] += X.transpose(-1, -2) @ true[t]
            if (nn_ + 1) % 20 == 0:
                log(f"fit {nn_ + 1}/{len(fit_items)} items")
        maps = {}
        for key, (xtx, xty) in acc.items():
            reg = RIDGE * torch.eye(xtx.shape[-1], dtype=torch.float64) * xtx.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
            maps[key] = torch.linalg.solve(xtx + reg, xty)
        del acc
        names = ["copy", "affine(copy)", "stream", "affine(stream)", "affine(stream+copy)"]
        err = {(kind, t): [torch.zeros(NL, dtype=torch.float64) for _ in range(len(names) + 1)] for kind in (0, 1) for t in range(len(TARGETS))}
        for i in test_items:
            f = features(i)
            if f is None:
                continue
            for kind, (true, copy, stream) in enumerate(f):
                for t in range(len(TARGETS)):
                    Y = true[t]
                    preds = [copy, design(copy, stream, "copy") @ maps[(kind, "copy", t)], stream,
                             design(copy, stream, "stream") @ maps[(kind, "stream", t)], design(copy, stream, "both") @ maps[(kind, "both", t)]]
                    e = err[(kind, t)]
                    for j, p in enumerate(preds):
                        e[j] += ((Y - p) ** 2).sum(dim=(1, 2, 3))
                    e[-1] += ((Y - Y.mean(dim=2, keepdim=True)) ** 2).sum(dim=(1, 2, 3))
    layers = [0, 1, 2, 4, 8, 16, 24, 32, 40, NL - 1]
    report = {}
    for kind, kname in ((0, "keys"), (1, "values")):
        for t, target in enumerate(TARGETS):
            e = err[(kind, t)]
            rel = [(x / e[-1]) for x in e[:-1]]
            report[f"{kname} 1->{target + 1}"] = {nm: [round(float(v), 4) for v in r] for nm, r in zip(names, rel)}
            log(f"{kname} loop 1 -> {target + 1}, mean over layers: " + "  ".join(f"{nm} {r.mean():.3f}" for nm, r in zip(names, rel)))
            for l in layers:
                log(f"    layer {l:2d}: " + "  ".join(f"{nm} {r[l]:.2f}" for nm, r in zip(names, rel)))
    json.dump({"fit_items": len(fit_items), "test_items": len(test_items), "report": report}, open(f"{WORK}/kv_stream_fit.json", "w"), indent=1)
    if save:
        torch.save({"targets": TARGETS, "maps": {f"{kind}:{t}": maps[(kind, "both", t)].float() for kind in (0, 1) for t in range(len(TARGETS))}, "layers": NL}, save)
        log(f"saved stream+copy maps to {save}")
    return report


if __name__ == "__main__":
    from bfcl_tf_exit import load_model
    args = [a for k, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[k - 1] != "--save"]
    save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    model, tok = load_model(os.environ.get("CKPT", "/home/jli199/boptim_scratch/ouro_base_exitlab"))
    fit(model, tok, int(args[0]) if args else 100, int(args[1]) if len(args) > 1 else 32, save, log=lambda s: print(s, flush=True))
