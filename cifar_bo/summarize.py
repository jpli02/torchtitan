"""Summarise the CIFAR-10 optimizer comparison of 2026-11-22 from the saved trajectories (one JSON per run, written after every
evaluation by trajectory_log.py): best test accuracy per run, mean and spread per arm, the seed-matched difference to
GP-hedge, and the mean best-so-far after k evaluations (k = 1..10). Runs still in progress are listed with their count.

  /home/jli199/boptim-agent/.venv/bin/python cifar_bo/summarize.py [root]      (root default: boptim_scratch/cifar_bo_1122)
Writes <root>/summary.json.
"""
import glob
import json
import os
import statistics as st
import sys
from math import comb, sqrt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/jli199/boptim_scratch/cifar_bo_1122"
# Earlier runs of the same setting (seeds 42, 43), read from their logs: logs/run_experiment_16_210856.log (GP-hedge),
# .claude/worktrees/cifar-save-trajectory/logs/run_experiment_17_172436.log (gpt-5.6-luna), logs/run_experiment_16_210919.log
# (qwen3-8b); gpt-5-mini and qwen3.5-9b as given in the user's table.
EARLIER = {"gp_hedge (Sep 16)": [0.9067, 0.9090], "gpt-5.6-luna (Sep 17)": [0.8977, 0.9010],
           "qwen3-8b (Sep 16)": [0.9088, 0.9083], "gpt-5-mini (user table)": [0.898, 0.899],
           "qwen3.5-9b (user table)": [0.90, 0.8966]}


def load(arm):
    runs = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "trajectories", arm, "*", "*.json"))):
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        vals = [float(o["value"]) for o in d.get("observations", [])]
        if not vals:
            continue
        seed = d.get("objective_seed", d.get("seed"))
        runs[seed] = {"values": vals, "best": max(vals), "n": len(vals), "status": d.get("status"),
                      "points": [o["point"] for o in d["observations"]], "file": p}
    return runs


arms = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "trajectories", "*")))
data = {a: load(a) for a in arms}
out = {"arms": {}, "earlier": EARLIER}
SEEDS = sorted({s for a in arms for s in data[a]})
print(f"{'arm':18s} " + "  ".join(f"{'s' + str(s):7s}" for s in SEEDS) + "    mean     sd   n_done")
for a in arms:
    r = data[a]
    cells = []
    for s in SEEDS:
        if s in r:
            cells.append(f"{r[s]['best']:.4f}" + ("" if r[s]["n"] >= 10 else f"({r[s]['n']})"))
        else:
            cells.append("   -   ")
    done = [r[s]["best"] for s in r if r[s]["n"] >= 10]
    mean = st.mean(done) if done else float("nan")
    sd = st.stdev(done) if len(done) > 1 else 0.0
    print(f"{a:18s} " + "  ".join(f"{c:7s}" for c in cells) + f"  {mean:.4f}  {sd:.4f}  {len(done)}")
    curve = []
    for k in range(1, 11):
        b = [max(r[s]["values"][:k]) for s in r if r[s]["n"] >= k]
        curve.append(round(st.mean(b), 4) if b else None)
    out["arms"][a] = {"per_seed": {str(s): {"best": r[s]["best"], "n": r[s]["n"], "values": r[s]["values"], "points": r[s]["points"]}
                                   for s in r}, "mean": mean, "sd": sd, "n_done": len(done), "best_so_far_mean": curve}

BASE = "gp_hedge" if "gp_hedge" in data else "gp10"  # the user's GP-hedge protocol (n_initial = budget = 10)
if BASE in data:
    print(f"\nseed-matched difference to {BASE} (finished runs only):")
    for a in arms:
        if a == BASE:
            continue
        common = [s for s in data[a] if s in data[BASE] and data[a][s]["n"] >= 10 and data[BASE][s]["n"] >= 10]
        diffs = [data[a][s]["best"] - data[BASE][s]["best"] for s in sorted(common)]
        if diffs:
            w, n = sum(d > 0 for d in diffs), sum(d != 0 for d in diffs)
            # exact two-sided sign test and the paired t statistic over seeds
            p_sign = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(w, n - w) + 1)) / 2 ** n) if n else 1.0
            sd_d = st.stdev(diffs) if len(diffs) > 1 else 0.0
            t = st.mean(diffs) / (sd_d / sqrt(len(diffs))) if sd_d > 0 else float("nan")
            print(f"  {a:18s} mean {st.mean(diffs):+.4f} over {len(diffs)} seeds, wins {w}/{len(diffs)}, "
                  f"sign p={p_sign:.3f}, paired t={t:+.2f}")
            print("      " + "  ".join(f"s{s}: {d:+.4f}" for s, d in zip(sorted(common), diffs)))
            out["arms"][a]["vs_" + BASE] = dict(zip(map(str, sorted(common)), diffs))

print("\nmean best-so-far after k evaluations:")
print("  k      " + "  ".join(f"{a[:14]:>14s}" for a in arms))
for k in range(10):
    print(f"  {k + 1:2d}     " + "  ".join(f"{(out['arms'][a]['best_so_far_mean'][k] or float('nan')):14.4f}" for a in arms))
if BASE == "gp_hedge":  # the September reference numbers belong to the 3-knob objective only
    print("\nearlier runs, same setting (seeds 42, 43): " + "; ".join(f"{k} {st.mean(v):.4f}" for k, v in EARLIER.items()))
json.dump(out, open(os.path.join(ROOT, "summary.json"), "w"), indent=1)
