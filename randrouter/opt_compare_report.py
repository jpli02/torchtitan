"""Report of randrouter/opt_compare.sh: best value so far after 5 / 10 / 20 / 30 evaluations per optimizer (mean and standard
error over seeds), how often one optimizer's run ends below another's (with a rank-sum test), how many evaluations a run needs
to get below the released single knob, and, because random search has no memory, its exact expected best-of-n from ALL of its
evaluations pooled (every n-subset of the pooled draws is a valid random-search run, so this needs no extra seeds).

  python randrouter/opt_compare_report.py [space ...]          (default: every space under the comparison root)
Values are minimised. Writes <root>/report.json.
"""
import glob
import itertools
import json
import math
import os
import statistics
import sys

ROOT = "/home/jli199/boptim_scratch/randrouter/opt_compare"
MARKS = (5, 10, 20, 30)
# the released model's single knob on the same proxy (randrouter/tf_bo_prep.py baselines): the value a search has to beat
BASELINE = {"thr": "/home/jli199/boptim_scratch/randrouter/bo_tf/base_copy/baseline.json",
            "nas6": "/home/jli199/boptim_scratch/randrouter/bo_tf/base_copy/baseline.json",     # router NAS: the released gate is the reference
            "nas8": "/home/jli199/boptim_scratch/randrouter/bo_tf/base_copy/baseline.json",
            "exact": "/home/jli199/boptim_scratch/randrouter/bo_tf/base_exact/baseline.json",
            "veto": "/home/jli199/boptim_scratch/randrouter/bo_tf/fill24/baseline.json"}


def load(space):
    """{optimizer: {seed: [values in evaluation order]}}"""
    out = {}
    for d in sorted(glob.glob(f"{ROOT}/{space}/*_s*/")):
        optim, seed = os.path.basename(d.rstrip("/")).rsplit("_s", 1)
        p = d + "evaluations.jsonl"
        if not os.path.exists(p):
            continue
        vals = [json.loads(l)["value"] for l in open(p) if l.strip()]
        if vals:
            out.setdefault(optim, {})[int(seed)] = vals
    return out


def best_so_far(vals):
    best, cur = [], float("inf")
    for v in vals:
        cur = min(cur, v)
        best.append(cur)
    return best


def expected_min(pool, n):
    """Exact expectation of the minimum of n draws without replacement from the pool."""
    xs = sorted(pool)
    N = len(xs)
    if n > N:
        return None
    tot = math.comb(N, n)
    return sum(x * math.comb(N - i - 1, n - 1) for i, x in enumerate(xs)) / tot


def quantile_min(pool, n, q):
    """Quantile q of the minimum of n draws without replacement: smallest x with P(min <= x) >= q."""
    xs = sorted(pool)
    N = len(xs)
    if n > N:
        return None
    tot = math.comb(N, n)
    for i, x in enumerate(xs):                      # P(min > x_i) = C(N-i-1, n) / C(N, n)
        if 1 - math.comb(N - i - 1, n) / tot >= q:
            return x
    return xs[-1]


def single_knob(space):
    p = BASELINE.get(space)
    if not p or not os.path.exists(p):
        return None
    return min(r["value"] for r in json.load(open(p))["single_knob"])


def rank_sum(a, b):
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return None


def report(space, log=print):
    runs = load(space)
    base = single_knob(space)
    res = {"space": space, "single_knob": base, "optimizers": {}, "pairs": [], "random_pooled": None}
    log(f"== {space}" + ("" if base is None else f"   (released single knob at its best setting: {base:.2f})"))
    for optim, rs in sorted(runs.items()):
        curves = {s: best_so_far(v) for s, v in rs.items()}
        row = {"runs": len(rs), "evaluations": sum(len(v) for v in rs.values())}
        cells = []
        for m in MARKS:
            xs = [b[m - 1] for b in curves.values() if len(b) >= m]
            if not xs:
                continue
            row[str(m)] = {"mean": statistics.mean(xs), "se": (statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else None),
                           "min": min(xs), "max": max(xs), "n": len(xs)}
            cells.append(f"{m}: {row[str(m)]['mean']:.2f}" + (f"±{row[str(m)]['se']:.2f}" if len(xs) > 1 else "") + f" [{min(xs):.2f}..{max(xs):.2f}]")
        if base is not None:      # evaluations until the run is below the released single knob (None = never within the run)
            first = [next((i + 1 for i, b in enumerate(c) if b < base - 1e-9), None) for c in curves.values()]
            row["first_below_single_knob"] = first
            hit = [f for f in first if f is not None]
            cells.append(f"below the single knob in {len(hit)}/{len(first)} runs" + (f" (median evaluation {statistics.median(hit):.0f})" if hit else ""))
        res["optimizers"][optim] = row
        log(f"  {optim:<9} {len(rs):>2} runs | best so far after " + "; ".join(cells))
    if "random" in runs:
        pool = [v for vs in runs["random"].values() for v in vs]
        rp = {"n_pool": len(pool)}
        cells = []
        for m in MARKS:
            e = expected_min(pool, m)
            if e is None:
                continue
            rp[str(m)] = {"mean": e, "q10": quantile_min(pool, m, 0.10), "median": quantile_min(pool, m, 0.5), "q90": quantile_min(pool, m, 0.90)}
            cells.append(f"{m}: {e:.2f} (median {rp[str(m)]['median']:.2f}, 10-90 % {rp[str(m)]['q10']:.2f}..{rp[str(m)]['q90']:.2f})")
        if base is not None:
            rp["share_below_single_knob"] = sum(v < base - 1e-9 for v in pool) / len(pool)
            cells.append(f"{100 * rp['share_below_single_knob']:.1f} % of single draws are below the single knob")
        res["random_pooled"] = rp
        log(f"  random, exact best-of-n from {len(pool)} pooled draws: " + "; ".join(cells))
    for a, b in itertools.combinations(sorted(runs), 2):
        for m in (10, 20, 30):
            fa = [best_so_far(v)[m - 1] for v in runs[a].values() if len(v) >= m]
            fb = [best_so_far(v)[m - 1] for v in runs[b].values() if len(v) >= m]
            if len(fa) < 2 or len(fb) < 2:
                continue
            wins = sum(x < y for x in fa for y in fb) + 0.5 * sum(x == y for x in fa for y in fb)
            p = rank_sum(fa, fb)
            res["pairs"].append({"a": a, "b": b, "after": m, "a_below_b": wins, "pairs": len(fa) * len(fb), "p": p})
            log(f"  after {m}: {a} below {b} in {wins:.0f} of {len(fa) * len(fb)} run pairs" + ("" if p is None else f" (rank-sum p = {p:.3f})"))
    return res


if __name__ == "__main__":
    spaces = sys.argv[1:] or sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(f"{ROOT}/*/"))
    allres = [report(s) for s in spaces]
    if allres:
        json.dump(allres, open(f"{ROOT}/report.json", "w"), indent=1)
