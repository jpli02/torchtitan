#!/usr/bin/env python3
"""Compare BO optimizers from raw trajectory dirs (gen_bo_traj.py output).

For every dir, per iteration t report the mean best-so-far value across
trajectories (minimize: lower is better), plus the final mean/median best and
how many trajectories failed to parse (missing files). Same seeds across dirs
make it a paired comparison on identical objective instances.

  python bo_compare.py --dirs gp=/path/gp_fourier2d qwen=/path/qwen-bo_fourier2d \
      --iters 15 [--seeds 1000-1019]
"""
import argparse
import glob
import json
import os
import statistics


def load(d):
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, "*.json"))):
        t = json.load(open(fp))
        out[t["seed"]] = t
    return out


def best_curve(t, iters):
    sign = -1.0 if t.get("direction", "minimize") == "maximize" else 1.0
    vals = [sign * o["value"] for o in t["observations"]]
    curve, best = [], float("inf")
    for i in range(iters):
        if i < len(vals):
            best = min(best, vals[i])
        curve.append(best)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="label=dir ...")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--seeds", default=None, help="a-b inclusive; default: seeds common to all dirs")
    a = ap.parse_args()
    runs = {}
    for spec in a.dirs:
        label, d = spec.split("=", 1)
        runs[label] = load(d)
    if a.seeds:
        lo, hi = map(int, a.seeds.split("-"))
        seeds = set(range(lo, hi + 1))
    else:
        seeds = set.intersection(*(set(r) for r in runs.values()))
    seeds = sorted(seeds)
    print(f"paired seeds: {len(seeds)}  iters: {a.iters}")
    marks = [0, 2, 4, 7, 9, 14]
    marks = [m for m in marks if m < a.iters]
    print("label      n   " + "  ".join(f"t={m + 1:<2d}   " for m in marks) + "  final mean  median")
    for label, r in runs.items():
        curves = [best_curve(r[s], a.iters) for s in seeds if s in r]
        n = len(curves)
        if not n:
            print(f"{label:10s} 0   (no trajectories for these seeds)")
            continue
        mean_t = [statistics.mean(c[m] for c in curves) for m in marks]
        finals = [c[-1] for c in curves]
        print(f"{label:10s} {n:<3d} " + "  ".join(f"{v:8.4f}" for v in mean_t)
              + f"  {statistics.mean(finals):10.4f}  {statistics.median(finals):7.4f}")
    # head-to-head wins on final best (first two labels)
    labels = list(runs)
    if len(labels) >= 2:
        a_, b_ = labels[0], labels[1]
        wins = ties = 0
        for s in seeds:
            if s in runs[a_] and s in runs[b_]:
                x, y = best_curve(runs[a_][s], a.iters)[-1], best_curve(runs[b_][s], a.iters)[-1]
                wins += x < y
                ties += x == y
        print(f"{a_} beats {b_} on {wins}/{len(seeds)} seeds (ties {ties})")


if __name__ == "__main__":
    main()
