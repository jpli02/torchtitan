"""What would a veto buy at equal loops? (Token-level arithmetic on the measured harm, before any decoding.)

For a share s of the validation answer tokens to release after loop 1, compare the measured harm of the released set when it
is chosen by
  gate        the s the released gate is surest about (what a threshold on the gate does);
  gate+veto   the gate's surest c (c > s), minus the ones the harm probe ranks most harmful, down to s;
  oracle      the gate's surest c minus the truly most harmful (upper bound for any probe at that c);
  random      a random s.
Harm is additive here (each token's integrated-gradients share), so this is a first-order estimate of the damage to the later
tokens; the real test is the teacher-forced router evaluation (tf_eval.py, "|veto=").

  python randrouter/harm_select.py --val harm_val.pt --probe DIR/linear.pt [--shares 0.2,0.3,0.4,0.5] [--cand 0.5,0.7,1.0]
"""
import argparse
import json
import os
import sys

import torch
from torch import nn

ap = argparse.ArgumentParser()
ap.add_argument("--val", required=True)
ap.add_argument("--probe", required=True)
ap.add_argument("--shares", default="0.2,0.3,0.4,0.5")
ap.add_argument("--cand", default="0.5,0.7,1.0")
ap.add_argument("--out", default=None)
args = ap.parse_args()

d = torch.load(args.val, map_location="cpu")
m = d["n_released"] > 0
x, h, lam = d["state1"][m].float(), d["harm_released"][m].float(), d["lam1"][m].float()
blob = torch.load(args.probe, map_location="cpu")
sd = blob["state_dict"]
net = (nn.Sequential(nn.Linear(sd["0.weight"].shape[1], sd["0.weight"].shape[0]), nn.GELU(), nn.Linear(sd["2.weight"].shape[1], 1))
       if blob["kind"] == "mlp" else nn.Linear(sd["weight"].shape[1], 1))
net.load_state_dict(sd)
with torch.no_grad():
    score = net((x - blob["mean"]) / blob["std"]).squeeze(-1)
n = h.numel()
gate_order = torch.argsort(lam, descending=True)
total = float(h.clamp(min=0).sum())
res = {"n_tokens": n, "mean_harm": float(h.mean()), "rows": []}
print(f"{n} validation tokens; mean harm {float(h.mean()):.4f}. Mean harm per RELEASED token (lower is better):")
print(f"{'released share':>15} {'random':>8} {'gate':>8} " + " ".join(f"{'gate ' + c + '+veto':>16} {'oracle':>8}" for c in args.cand.split(",")))
for s in [float(v) for v in args.shares.split(",")]:
    k = int(n * s)
    row = {"share": s, "random": float(h.mean()), "gate": float(h[gate_order[:k]].mean()), "veto": {}, "oracle": {}}
    cells = []
    for c in [float(v) for v in args.cand.split(",")]:
        if c <= s:
            cells.append(f"{'—':>16} {'—':>8}")
            continue
        cand = gate_order[:int(n * c)]
        keep = cand[torch.argsort(score[cand])[:k]]                    # the k candidates the probe scores least harmful
        best = cand[torch.argsort(h[cand])[:k]]
        row["veto"][str(c)], row["oracle"][str(c)] = float(h[keep].mean()), float(h[best].mean())
        cells.append(f"{row['veto'][str(c)]:>16.4f} {row['oracle'][str(c)]:>8.4f}")
    res["rows"].append(row)
    print(f"{s:>15.2f} {row['random']:>8.4f} {row['gate']:>8.4f} " + " ".join(cells))
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
