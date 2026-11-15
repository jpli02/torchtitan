"""How good a stand-in is a token's loop-1 key/value for the key/value the deeper loops would have computed, and how much of the
gap does a linear correction close?

Real early exit fills an exited token's cache slots of the skipped loops with copies of its exit loop's keys and values. This
script measures, on full-depth answers (teacher-forced, cache kept), for every layer and for source loop -> target loop pairs
(1->2, 1->3, 1->4 and, as the benign reference, 2->3, 2->4):
  * copy error:   |target - source|^2 / |target - mean(target)|^2 on held-out tokens (1.0 = no better than a constant);
  * affine error: the same after target ~ source @ A + b, one (A, b) per layer, head, loop pair and for keys and values
                  separately, fitted by ridge regression on the answer tokens of the fit items.
With --save the fitted maps are written to <out>.pt for the decode-time fill (boptim_lcb/make_calfill_modeling.py).

  CUDA_VISIBLE_DEVICES=0 python boptim_lcb/kv_fill_fit.py [n_fit] [n_test] [--save path]
"""
import glob
import json
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

CKPT = "/home/jli199/boptim_scratch/ouro_base_kvshare"
DATA = "/srv/scratch/jli199/bfcl_stage/bfcl_search.jsonl"
WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
N_FIT = int(args[0]) if len(args) > 0 else 120
N_TEST = int(args[1]) if len(args) > 1 else 40
SAVE = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # (source loop, target loop), zero-based
RIDGE = 1e-2

rows = [json.loads(l) for l in open(DATA) if l.strip()]
full_dir = None
for rec in sorted(glob.glob(f"{WORK}/q1.000-1.000-1.000_*/record.json")):
    if json.load(open(rec)).get("status") == "ok":
        full_dir = rec.rsplit("/", 1)[0]
gens = [json.loads(l) for l in open(f"{full_dir}/gen.jsonl") if l.strip() and '"summary"' not in l[:12]]
step = max(1, len(rows) // (N_FIT + N_TEST))
pick = list(range(0, len(rows), step))[: N_FIT + N_TEST]
fit_items, test_items = [], []
for j, i in enumerate(pick):                                  # every fourth picked item is a test item
    (test_items if j % 4 == 3 and len(test_items) < N_TEST else fit_items).append(i)

tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
config = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
config.total_ut_steps = 4
model = AutoModelForCausalLM.from_pretrained(CKPT, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
NL = config.num_hidden_layers


def answer_kv(i):
    """keys, values of the answer tokens: [loops, layers, heads, n_answer_tokens, dim] each (float64, CPU)."""
    r, g = rows[i], gens[i]
    text = tok.apply_chat_template([{"role": "user", "content": r["prompt"].rstrip()}], tokenize=False, add_generation_prompt=True) + "["
    p_ids = tok(text)["input_ids"]
    a_ids = tok(g["gen"][1:] if g["gen"].startswith("[") else g["gen"], add_special_tokens=False)["input_ids"]
    if not a_ids:
        return None
    ids = torch.tensor([p_ids + a_ids], device="cuda")
    out = model(input_ids=ids, use_cache=True)
    pk = out.past_key_values
    lo = len(p_ids)
    # cached keys carry the rotary rotation of their position; the linear relation between loops lives in the unrotated space
    # (a map fitted on rotated keys would have to commute with every rotation), so rotate the keys back first
    cos, sin = model.model.rotary_emb(torch.zeros(1, device="cuda", dtype=torch.float32), torch.arange(ids.shape[1], device="cuda")[None])
    cos, sin = cos[0, lo:].float(), sin[0, lo:].float()                       # [n_answer_tokens, dim]

    def unrot(k):                                                             # [heads, n, dim], inverse of apply_rotary_pos_emb
        k = k.float()
        half = k.shape[-1] // 2
        return k * cos - torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin

    K = torch.stack([torch.stack([unrot(pk.key_cache[s * NL + l][0, :, lo:, :]) for l in range(NL)]) for s in range(4)]).double().cpu()
    V = torch.stack([torch.stack([pk.value_cache[s * NL + l][0, :, lo:, :].float() for l in range(NL)]) for s in range(4)]).double().cpu()
    return K, V


acc = None
with torch.inference_mode():
    for n, i in enumerate(fit_items):
        kv = answer_kv(i)
        if kv is None:
            continue
        for which, T in enumerate(kv):                        # 0 keys, 1 values
            H, D = T.shape[2], T.shape[4]
            if acc is None:
                acc = {(w, p): [torch.zeros(NL, H, D + 1, D + 1, dtype=torch.float64), torch.zeros(NL, H, D + 1, D, dtype=torch.float64)]
                       for w in (0, 1) for p in PAIRS}
            ones = torch.ones(*T.shape[1:4], 1, dtype=torch.float64)
            for (a, b) in PAIRS:
                X = torch.cat([T[a], ones], dim=-1)           # [layers, heads, n, D+1]
                acc[(which, (a, b))][0] += X.transpose(-1, -2) @ X
                acc[(which, (a, b))][1] += X.transpose(-1, -2) @ T[b]
        if (n + 1) % 20 == 0:
            print(f"fit {n + 1}/{len(fit_items)} items", flush=True)
    maps = {}
    for key, (xtx, xty) in acc.items():
        reg = RIDGE * torch.eye(xtx.shape[-1], dtype=torch.float64) * xtx.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
        maps[key] = torch.linalg.solve(xtx + reg, xty)        # [layers, heads, D+1, D]
    # held-out error
    err = {key: [torch.zeros(NL, dtype=torch.float64), torch.zeros(NL, dtype=torch.float64), torch.zeros(NL, dtype=torch.float64)] for key in maps}
    for n, i in enumerate(test_items):
        kv = answer_kv(i)
        if kv is None:
            continue
        for which, T in enumerate(kv):
            ones = torch.ones(*T.shape[1:4], 1, dtype=torch.float64)
            for (a, b) in PAIRS:
                Y = T[b]
                pred = torch.cat([T[a], ones], dim=-1) @ maps[(which, (a, b))]
                err[(which, (a, b))][0] += ((Y - T[a]) ** 2).sum(dim=(1, 2, 3))
                err[(which, (a, b))][1] += ((Y - pred) ** 2).sum(dim=(1, 2, 3))
                err[(which, (a, b))][2] += ((Y - Y.mean(dim=2, keepdim=True)) ** 2).sum(dim=(1, 2, 3))
report = {}
layers = [0, 1, 2, 4, 8, 16, 24, 32, 40, NL - 1]
for which, name in ((0, "keys"), (1, "values")):
    for (a, b) in PAIRS:
        c, f, t = err[(which, (a, b))]
        copy_e, aff_e = (c / t), (f / t)
        report[f"{name} {a + 1}->{b + 1}"] = {"copy": [round(float(v), 4) for v in copy_e], "affine": [round(float(v), 4) for v in aff_e]}
        print(f"{name} loop {a + 1} -> {b + 1}: relative error, copy / affine, mean over layers {copy_e.mean():.3f} / {aff_e.mean():.3f}; by layer "
              + "  ".join(f"L{l}: {copy_e[l]:.2f}/{aff_e[l]:.2f}" for l in layers if l < NL), flush=True)
json.dump({"fit_items": len(fit_items), "test_items": len(test_items), "ridge": RIDGE, "report": report}, open(f"{WORK}/kv_fill_fit.json", "w"), indent=1)
if SAVE:
    torch.save({"pairs": PAIRS, "maps": {f"{w}:{a}:{b}": m.float() for (w, (a, b)), m in maps.items()}, "layers": NL}, SAVE)
    print("saved maps to", SAVE)
print("wrote", f"{WORK}/kv_fill_fit.json")
