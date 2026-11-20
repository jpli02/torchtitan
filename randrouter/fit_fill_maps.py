"""Calibrated cache fill for LiveCodeBench answers, with no training of the model: for every layer, head and pair of loops
(exit loop a -> skipped loop b) an affine map of a token's loop-a keys (unrotated) and values onto its loop-b ones, fitted by
ridge regression on the answer tokens of teacher answers (full depth, exact cache). The same fit and file format as
boptim_lcb/kv_fill_fit.py (BFCL), so the exit-lab serving model (decode_fill_maps) and core.routed_forward (fill_maps) both
read it. Reports, on held-out answers, the relative error of the plain copy and of the map per pair.

  CUDA_VISIBLE_DEVICES=0 python randrouter/fit_fill_maps.py --save FILE.pt [--n_fit 120] [--n_test 20] [--split train]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
RIDGE = 1e-2

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl")
ap.add_argument("--split", default="train")
ap.add_argument("--n_fit", type=int, default=120)
ap.add_argument("--n_test", type=int, default=20)
ap.add_argument("--max_len", type=int, default=1500)
ap.add_argument("--save", required=True)
ap.add_argument("--mem_gb", type=float, default=0.0)
args = ap.parse_args()
if args.mem_gb:
    torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
model, tok = core.load_model(core.SERVE_DIR)
NL = model.config.num_hidden_layers
seqs = [s for s in core.load_teacher(args.teacher, split=args.split) if len(s["ids"]) <= args.max_len]
fit_items, test_items = seqs[:args.n_fit], seqs[args.n_fit:args.n_fit + args.n_test]
print(f"fit on {len(fit_items)} answers, test on {len(test_items)} ({args.split}, at most {args.max_len} tokens)", flush=True)


def answer_kv(s):
    """Unrotated keys and values of the decoded positions: [loops, layers, heads, n, dim] each (float64, CPU)."""
    ids = torch.tensor([s["ids"]], device="cuda")
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    pk = out.past_key_values
    lo = s["n_prompt"]
    cos, sin = model.model.rotary_emb(torch.zeros(1, device="cuda", dtype=torch.float32), torch.arange(ids.shape[1], device="cuda")[None])
    cos, sin = cos[0, lo:].float(), sin[0, lo:].float()

    def unrot(k):
        k = k.float()
        half = k.shape[-1] // 2
        return k * cos - torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin

    K = torch.stack([torch.stack([unrot(pk.key_cache[t * NL + l][0, :, lo:, :]) for l in range(NL)]) for t in range(4)]).double().cpu()
    V = torch.stack([torch.stack([pk.value_cache[t * NL + l][0, :, lo:, :].float() for l in range(NL)]) for t in range(4)]).double().cpu()
    del pk, out
    return K, V


acc = None
with torch.inference_mode():
    for n, s in enumerate(fit_items):
        for which, T in enumerate(answer_kv(s)):
            H, D = T.shape[2], T.shape[4]
            if acc is None:
                acc = {(w, p): [torch.zeros(NL, H, D + 1, D + 1, dtype=torch.float64), torch.zeros(NL, H, D + 1, D, dtype=torch.float64)]
                       for w in (0, 1) for p in PAIRS}
            ones = torch.ones(*T.shape[1:4], 1, dtype=torch.float64)
            for (a, b) in PAIRS:
                X = torch.cat([T[a], ones], dim=-1)
                acc[(which, (a, b))][0] += X.transpose(-1, -2) @ X
                acc[(which, (a, b))][1] += X.transpose(-1, -2) @ T[b]
        if (n + 1) % 10 == 0:
            print(f"fit {n + 1}/{len(fit_items)}", flush=True)
    maps = {}
    for key, (xtx, xty) in acc.items():
        reg = RIDGE * torch.eye(xtx.shape[-1], dtype=torch.float64) * xtx.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
        maps[key] = torch.linalg.solve(xtx + reg, xty)
    err = {key: [torch.zeros(NL, dtype=torch.float64) for _ in range(3)] for key in maps}
    for s in test_items:
        for which, T in enumerate(answer_kv(s)):
            ones = torch.ones(*T.shape[1:4], 1, dtype=torch.float64)
            for (a, b) in PAIRS:
                Y = T[b]
                pred = torch.cat([T[a], ones], dim=-1) @ maps[(which, (a, b))]
                err[(which, (a, b))][0] += ((Y - T[a]) ** 2).sum(dim=(1, 2, 3))
                err[(which, (a, b))][1] += ((Y - pred) ** 2).sum(dim=(1, 2, 3))
                err[(which, (a, b))][2] += ((Y - Y.mean(dim=2, keepdim=True)) ** 2).sum(dim=(1, 2, 3))
report = {}
for which, name in ((0, "keys"), (1, "values")):
    for (a, b) in PAIRS:
        c, f, t = err[(which, (a, b))]
        report[f"{name} {a + 1}->{b + 1}"] = {"copy": [round(float(v), 4) for v in c / t], "affine": [round(float(v), 4) for v in f / t]}
        print(f"{name} loop {a + 1} -> {b + 1}: relative error copy / affine, mean over layers {float((c / t).mean()):.3f} / {float((f / t).mean()):.3f}", flush=True)
torch.save({"pairs": PAIRS, "maps": {f"{w}:{a}:{b}": m.float() for (w, (a, b)), m in maps.items()}, "layers": NL}, args.save)
json.dump({"fit_items": len(fit_items), "test_items": len(test_items), "ridge": RIDGE, "report": report}, open(args.save + ".report.json", "w"), indent=1)
print(f"saved {args.save}", flush=True)
