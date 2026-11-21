"""How sure is the calibration of calib_pass1.py? Bootstrap over the real runs (resampled with replacement, refitted), and what
that uncertainty means for the routers the calibration recommends: for every resample, the predicted pass@1 and objective value
of the released gate at 0.3 (plain copy) and of calibrated-fill routers, and how often the calibrated fill comes out ahead.
Two noise sources are covered: which routers happen to be in the calibration set (resampling) and the binomial noise of every
real pass@1 (each resampled measurement is redrawn from Binomial(n, p)). CPU only.

  python randrouter/calib_boot.py [--draws 4000]
"""
import argparse
import json

import numpy as np

R = "/home/jli199/boptim_scratch/randrouter"


def fit(a, b, y, w):
    def solve(cols):
        X = np.stack([np.ones_like(y)] + [-c for c in cols], axis=1) * np.sqrt(w)[:, None]
        return np.linalg.lstsq(X, y * np.sqrt(w), rcond=None)[0]
    c, al, be = solve([a, b])
    if al < 0 and be < 0:
        return float(np.average(y, weights=w)), 0.0, 0.0
    if al < 0:
        c, be = solve([b])
        return float(c), 0.0, float(max(be, 0.0))
    if be < 0:
        c, al = solve([a])
        return float(c), float(max(al, 0.0)), 0.0
    return float(c), float(al), float(be)


def value(loops, own, cache, m):
    p = min(m[0], max(0.0, m[0] - m[1] * own - m[2] * cache))
    return -6.25 * 4 / loops + 2.0 * max(0.0, 18.3 - p), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=4000)
    a_ = ap.parse_args()
    cal = json.load(open(f"{R}/calib_pass1.json"))
    pts = cal["points"]
    a = np.array([p["a"] for p in pts])
    b = np.array([p["b"] for p in pts])
    y = np.array([p["pass1"] for p in pts])
    n = np.array([p["n"] for p in pts])
    w = n / 342.0
    m0 = (cal["model"]["c"], cal["model"]["alpha"], cal["model"]["beta"])
    cand = []
    for tab, spec, name in [("base", "0.3", "plain copy, gate 0.3"), ("fill24", "0.2,0.4,0.7", "calibrated fill (0.2, 0.4, 0.7)"),
                            ("fill24", "0.245,0.45,0.7", "calibrated fill (0.245, 0.45, 0.7)"), ("fill24", "0.245,0.3,0.3", "calibrated fill (0.245, 0.3, 0.3)"),
                            ("fill24", "0.2", "calibrated fill, gate 0.2")]:
        r = json.load(open(f"{R}/tables/{tab}.json"))["routers"][spec]
        cand.append((name, r["loops"], 100 - 100 * r["agree_exact"], 100 * (r["agree_exact"] - r["agree_real"])))
    rng = np.random.default_rng(0)
    ms, vals, ps = [], [], []
    for _ in range(a_.draws):
        idx = rng.integers(0, len(pts), len(pts))
        yb = 100.0 * rng.binomial(n[idx], np.clip(y[idx] / 100.0, 0, 1)) / n[idx]
        m = fit(a[idx], b[idx], yb, w[idx])
        ms.append(m)
        vp = [value(l, o, c, m) for _, l, o, c in cand]
        vals.append([v for v, _ in vp])
        ps.append([p for _, p in vp])
    ms, vals, ps = np.array(ms), np.array(vals), np.array(ps)
    q = lambda x: np.percentile(x, [2.5, 50, 97.5])  # noqa: E731
    print(f"fit on all runs: c {m0[0]:.2f}, alpha {m0[1]:.2f}, beta {m0[2]:.2f}")
    for k, name in enumerate(("c", "alpha (per % of own errors)", "beta (per % of cache errors)")):
        lo, md, hi = q(ms[:, k])
        print(f"  {name:<30} median {md:.2f}, 95 % interval {lo:.2f} to {hi:.2f}")
    r = ms[:, 1] / np.maximum(ms[:, 2], 1e-6)
    print(f"  alpha / beta                   median {np.median(r):.1f}, 95 % interval {np.percentile(r, 2.5):.1f} to {np.percentile(r, 97.5):.1f}; "
          f"alpha > beta in {100 * (ms[:, 1] > ms[:, 2]).mean():.0f} % of the resamples")
    out = {"draws": a_.draws, "coef": {"c": q(ms[:, 0]).tolist(), "alpha": q(ms[:, 1]).tolist(), "beta": q(ms[:, 2]).tolist()}, "candidates": []}
    for k, (name, l, o, c) in enumerate(cand):
        v0, p0 = value(l, o, c, m0)
        lo, md, hi = q(vals[:, k])
        plo, pmd, phi = q(ps[:, k])
        ahead = float((vals[:, k] < vals[:, 0]).mean()) if k else None
        print(f"  {name:<36} loops {l:.2f} own {o:.2f} cache {c:.2f}: pass@1 {p0:.1f} ({plo:.1f} to {phi:.1f}), value {v0:.2f} ({lo:.2f} to {hi:.2f})"
              + ("" if ahead is None else f"; below the released gate in {100 * ahead:.0f} % of the resamples, median margin {np.median(vals[:, 0] - vals[:, k]):.2f}"))
        out["candidates"].append({"name": name, "loops": l, "own": o, "cache": c, "pass1": p0, "pass1_ci": [plo, phi], "value": v0, "value_ci": [lo, hi], "ahead": ahead})
    json.dump(out, open(f"{R}/calib_boot.json", "w"), indent=1)


if __name__ == "__main__":
    main()
