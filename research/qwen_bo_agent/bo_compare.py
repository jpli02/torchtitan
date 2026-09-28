#!/usr/bin/env python3
"""Compare BO optimizers from raw trajectory dirs (research/qwen_bo_agent/gen_bo_traj.py output).

For every dir, per iteration t report the mean best-so-far value across
trajectories (minimize: lower is better), plus the final mean/median best and
how many trajectories failed to parse (missing files). Same seeds across dirs
make it a paired comparison on identical objective instances.

  python research/qwen_bo_agent/bo_compare.py --dirs gp=/path/gp_fourier2d qwen=/path/qwen-bo_fourier2d \
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


def best_curve(t, iters, norm=None):
    """Best-so-far per iteration; with norm=(lo, hi) returns normalised regret
    (best - lo) / (hi - lo) in [0, 1], 0 = grid optimum found."""
    sign = -1.0 if t.get("direction", "minimize") == "maximize" else 1.0
    vals = [sign * o["value"] for o in t["observations"]]
    curve, best = [], float("inf")
    for i in range(iters):
        if i < len(vals):
            best = min(best, vals[i])
        curve.append(best if norm is None else (best - norm[0]) / (norm[1] - norm[0]))
    return curve


def grid_range(objective, seed, n_grid=200):
    """(min, max) of the objective on an n_grid^2 grid -- fourier2d & co. are
    cheap closed forms, so this is a usable proxy for the global optimum."""
    import sys
    from types import SimpleNamespace

    import numpy as np

    sys.path.insert(0, os.environ.get("BOPTIM_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
    from objective import build_objective

    a = SimpleNamespace(objective=objective, objective_seed=seed, seed=seed, n_dims=None, n_fourier=5,
                        bounds=None, direction="minimize", epochs=1, cifar_metric="test_accuracy",
                        metrics=None, train_batch_size=128)
    obj = build_objective(a)
    b = obj.bounds
    if len(b) != 2:
        return None
    xs = np.linspace(b[0][0], b[0][1], n_grid)
    ys = np.linspace(b[1][0], b[1][1], n_grid)
    v = np.array([[obj.evaluate([x, y]) for y in ys] for x in xs])
    return float(v.min()), float(v.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="label=dir ...")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--seeds", default=None, help="a-b inclusive; default: seeds common to all dirs")
    ap.add_argument("--normalize", default=None, metavar="OBJECTIVE",
                    help="report normalised regret against the objective's 200x200 grid min/max per seed")
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
    norms = {}
    if a.normalize:
        for s in seeds:
            norms[s] = grid_range(a.normalize, s)
        print(f"normalised regret vs grid optimum of {a.normalize} (0 = optimum found)")
    print(f"paired seeds: {len(seeds)}  iters: {a.iters}")
    marks = [0, 2, 4, 7, 9, 14]
    marks = [m for m in marks if m < a.iters]
    print("label      n   " + "  ".join(f"t={m + 1:<2d}   " for m in marks) + "  final mean  median")
    for label, r in runs.items():
        curves = [best_curve(r[s], a.iters, norms.get(s)) for s in seeds if s in r]
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
                x = best_curve(runs[a_][s], a.iters, norms.get(s))[-1]
                y = best_curve(runs[b_][s], a.iters, norms.get(s))[-1]
                wins += x < y
                ties += x == y
        print(f"{a_} beats {b_} on {wins}/{len(seeds)} seeds (ties {ties})")


if __name__ == "__main__":
    main()
