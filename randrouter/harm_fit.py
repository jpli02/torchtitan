"""Stage II with the externality: fit a read-out of the loop-1 state that predicts what releasing the token after loop 1 costs
the LATER tokens (harm.py), i.e. the half of the exit decision the released gate was never trained on.

Probes (all read the 2,048-dim normed loop-1 state, like the released gate):
  linear   logistic regression, target "harm in the top --top share of the training tokens"
  mlp      one hidden layer of 256 units, same target
  reg      linear regression on log(1 + harm / scale)
Reported on held-out answers: AUC for the top-share target, and the share of the measured harm that sits in the tokens each
probe ranks as its most harmful 10 / 20 / 30 % (a perfect ranking and the released gate's own ranking are printed next to it).

  python randrouter/harm_fit.py --train harm_train.pt --val harm_val.pt --out DIR [--top 0.2] [--epochs 30]
Writes <out>/{linear,mlp,reg}.pt ({"kind", "state_dict", "mean", "std"}) and report.json.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_signal import auc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--val", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--top", type=float, default=0.2)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load(path):
    d = torch.load(path, map_location="cpu")
    m = d["n_released"] > 0
    return {"x": d["state1"][m].float(), "h": d["harm_released"][m].float(), "lam": d["lam1"][m].float(), "ok1": d["ok1"][m]}


tr, va = load(args.train), load(args.val)
mean, std = tr["x"].mean(0, keepdim=True), tr["x"].std(0, keepdim=True).clamp(min=1e-3)
cut = torch.quantile(tr["h"], 1 - args.top)
scale = tr["h"].clamp(min=0).mean().clamp(min=1e-6)
print(f"train {tr['h'].numel()} tokens, val {va['h'].numel()}; harm mean {float(tr['h'].mean()):.4f}, top-{round(100 * args.top)} % threshold {float(cut):.4f}", flush=True)


def norm(x):
    return ((x - mean) / std).to(dev)


def make(kind):
    if kind == "mlp":
        return nn.Sequential(nn.Linear(tr["x"].shape[1], 256), nn.GELU(), nn.Linear(256, 1)).to(dev)
    return nn.Linear(tr["x"].shape[1], 1).to(dev)


def captured(score, harm, shares=(0.1, 0.2, 0.3)):
    """Share of the positive harm in the tokens with the highest score."""
    pos = harm.clamp(min=0)
    order = torch.argsort(score, descending=True)
    c = torch.cumsum(pos[order], 0) / pos.sum().clamp(min=1e-9)
    return [float(c[max(1, int(len(order) * s)) - 1]) for s in shares]


report = {"top": args.top, "n_train": int(tr["h"].numel()), "n_val": int(va["h"].numel()), "probes": {}}
xtr, xva = norm(tr["x"]), norm(va["x"])
ytr_cls, yva_cls = (tr["h"] >= cut).float().to(dev), (va["h"] >= cut)
ytr_reg = torch.log1p(tr["h"].clamp(min=0) / scale).to(dev)
for kind in ("linear", "mlp", "reg"):
    net = make(kind)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    n = xtr.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, 1024):
            idx = perm[s:s + 1024]
            out = net(xtr[idx]).squeeze(-1)
            loss = F.mse_loss(out, ytr_reg[idx]) if kind == "reg" else F.binary_cross_entropy_with_logits(out, ytr_cls[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        sv = net(xva).squeeze(-1).float().cpu()
        st = net(xtr).squeeze(-1).float().cpu()
    rep = {"auc_val": auc(sv, yva_cls), "auc_train": auc(st, tr["h"] >= cut), "captured_val_10_20_30": captured(sv, va["h"])}
    report["probes"][kind] = rep
    torch.save({"kind": kind, "state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "mean": mean, "std": std,
                "score_quantiles_val": [float(torch.quantile(sv, q)) for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]}, os.path.join(args.out, f"{kind}.pt"))
    print(f"{kind:<7} AUC val {rep['auc_val']:.3f} (train {rep['auc_train']:.3f}); harm in its top 10/20/30 %: "
          f"{[round(100 * v) for v in rep['captured_val_10_20_30']]} %", flush=True)
# references on the validation tokens: a perfect ranking, the released gate (least sure = most harmful), "readout wrong"
refs = {"perfect ranking": va["h"], "released gate (least sure first)": -va["lam"], "loop-1 readout wrong": (~va["ok1"]).float()}
for name, sc in refs.items():
    c = captured(sc, va["h"])
    report["probes"][name] = {"auc_val": auc(sc, yva_cls), "captured_val_10_20_30": c}
    print(f"{name:<34} AUC {report['probes'][name]['auc_val']:.3f}; harm in its top 10/20/30 %: {[round(100 * v) for v in c]} %", flush=True)
json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)
