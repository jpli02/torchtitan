"""Is the cache stand-in good where it is used? The errors of kv_fill_fit.py average over all answer tokens, but a router only
releases the tokens it is confident about. This splits the held-out error of the copy and of the affine stand-in (loop 1 -> 2, 3, 4)
by the head's confidence after loop 1: tokens a confidence router at 0.99 would release, and the rest. Errors are relative to
the variance of the target over all tokens, by layer band (lower / middle / upper third).

  CUDA_VISIBLE_DEVICES=0 python boptim_lcb/kv_fill_split.py [n_items]        (items 0, 4, 8, ...: never used to fit the maps)
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_tf_exit import load_model  # noqa: E402

DATA = "/srv/scratch/jli199/bfcl_stage/bfcl_search.jsonl"
WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
TAU = 0.99

rows = [json.loads(l) for l in open(DATA) if l.strip()]
sig = [json.loads(l) for l in open(f"{WORK}/signal_probs.jsonl")]
blob = torch.load(f"{WORK}/fill_maps.pt", map_location="cpu")
maps = {tuple(int(x) for x in k.split(":")): v.double() for k, v in blob["maps"].items()}     # (kind, a, b) -> [layers, heads, D+1, D]
model, tok = load_model("/home/jli199/boptim_scratch/ouro_base_exitlab")
inner = model.model
NL = model.config.num_hidden_layers
bands = [("layers 0-15", 0, 16), ("layers 16-31", 16, 32), ("layers 32-47", 32, 48)]
# sums[(kind, target, band, group)] = [copy err, affine err, n tokens]; var[(kind, target, band)] = [sum sq dev, n]
sums, var = {}, {}
items = list(range(0, len(rows), 4))[:: max(1, len(range(0, len(rows), 4)) // N)][:N]
with torch.inference_mode():
    for n, i in enumerate(items):
        r, e = rows[i], sig[i]
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"].rstrip()}], tokenize=False, add_generation_prompt=True) + "["
        p_ids = tok(text)["input_ids"]
        a_ids = e["target"][:-1]
        if not a_ids:
            continue
        ids = torch.tensor([p_ids + a_ids], device="cuda")
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        pk = out.past_key_values
        lo = len(p_ids)
        cos, sin = inner.rotary_emb(torch.zeros(1, device="cuda", dtype=torch.float32), torch.arange(ids.shape[1], device="cuda")[None])
        cos, sin = cos[0, lo:].float(), sin[0, lo:].float()

        def unrot(k):
            k = k.float()
            half = k.shape[-1] // 2
            return k * cos - torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin

        # answer token j is fed at position lo + j; the exit decision for it uses the head after it has been processed, i.e. the
        # confidence recorded for the NEXT prediction, index j + 1 in the signal dump
        conf1 = torch.tensor(e["conf"][0][1:len(a_ids) + 1])
        grp = conf1 >= TAU
        for kind in (0, 1):
            cache = pk.key_cache if kind == 0 else pk.value_cache
            get = (lambda s, l: unrot(cache[s * NL + l][0, :, lo:, :])) if kind == 0 else (lambda s, l: cache[s * NL + l][0, :, lo:, :].float())
            src = torch.stack([get(0, l) for l in range(NL)]).double().cpu()                      # [layers, heads, n, D]
            ones = torch.ones(*src.shape[:3], 1, dtype=torch.float64)
            for b in (1, 2, 3):
                Y = torch.stack([get(b, l) for l in range(NL)]).double().cpu()
                pred = torch.cat([src, ones], dim=-1) @ maps[(kind, 0, b)]
                ce = ((Y - src) ** 2).sum(dim=(1, 3))                                             # [layers, n]
                ae = ((Y - pred) ** 2).sum(dim=(1, 3))
                dv = ((Y - Y.mean(dim=2, keepdim=True)) ** 2).sum(dim=(1, 3))
                for name, l0, l1 in bands:
                    v = var.setdefault((kind, b, name), [0.0, 0])
                    v[0] += float(dv[l0:l1].sum())
                    v[1] += Y.shape[2] * (l1 - l0)
                    for gname, m in (("released (confidence >= 0.99)", grp), ("kept", ~grp)):
                        s = sums.setdefault((kind, b, name, gname), [0.0, 0.0, 0])
                        s[0] += float(ce[l0:l1][:, m].sum())
                        s[1] += float(ae[l0:l1][:, m].sum())
                        s[2] += int(m.sum()) * (l1 - l0)
        del pk, out
        if (n + 1) % 16 == 0:
            print(f"{n + 1}/{len(items)} items", flush=True)
report = {}
for (kind, b, name, gname), (ce, ae, cnt) in sorted(sums.items()):
    v = var[(kind, b, name)]
    per_tok_var = v[0] / max(1, v[1])
    key = f"{'keys' if kind == 0 else 'values'} 1->{b + 1} {name} {gname}"
    report[key] = {"copy": round(ce / max(1, cnt) / per_tok_var, 3), "affine": round(ae / max(1, cnt) / per_tok_var, 3), "token_layers": cnt}
    print(f"{key}: copy {report[key]['copy']:.2f}  affine {report[key]['affine']:.2f}")
json.dump(report, open(f"{WORK}/kv_fill_split.json", "w"), indent=1)
print("wrote", f"{WORK}/kv_fill_split.json")
