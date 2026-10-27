"""Per-epoch training-loss statistics from offline wandb runs.

Usage: python wandb_epoch_stats.py NAME=PATH.wandb:STEPS_PER_EPOCH [...]
Prints the logged keys once, then per run: mean/min loss per epoch, last-10% mean,
and the fraction of logged steps with loss < 0.05 (memorised) in the final epoch.
"""
import statistics
import sys

from extract_wandb_loss import read_history

LOSS = "loss_metrics/global_avg_loss"
first = True
for spec in sys.argv[1:]:
    name, rest = spec.split("=", 1)
    path, spe = rest.rsplit(":", 1)
    spe = int(spe)
    rows, keys = read_history(path)
    if first:
        print("logged keys:", ", ".join(sorted(k for k in keys if not k.startswith("_"))))
        first = False
    pts = [(int(r["_step"]), float(r[LOSS])) for r in rows if r.get("_step") is not None and LOSS in r]
    pts.sort()
    last = pts[-1][0]
    n_ep = max(1, round(last / spe))
    print(f"\n{name}: {len(pts)} logged steps, last step {last}, {spe} steps/epoch -> {n_ep} epochs")
    other = {k: rows[-1].get(k) for k in ("optimizer/lr", "lr", "grad_norm", "optimizer/grad_norm") if k in rows[-1]}
    if other:
        print("  final:", other)
    for e in range(n_ep):
        seg = [l for s, l in pts if e * spe < s <= (e + 1) * spe]
        if not seg:
            continue
        print(f"  epoch {e+1}: mean {statistics.mean(seg):.3f}  median {statistics.median(seg):.3f}  "
              f"min {min(seg):.3f}  max {max(seg):.3f}  frac<0.05 {sum(l < 0.05 for l in seg)/len(seg):.0%}  (n={len(seg)})")
    tail = [l for s, l in pts if s > last * 0.9]
    print(f"  last 10% of steps: mean {statistics.mean(tail):.3f}, median {statistics.median(tail):.3f}")
