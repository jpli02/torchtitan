"""Report of randrouter/opt_compare.sh: best value so far after 5 / 10 / 20 / 30 evaluations per optimizer (mean and standard
error over seeds), and how often one optimizer's run ends below another's.

  python randrouter/opt_compare_report.py [space ...]          (default: every space under the comparison root)
"""
import glob
import itertools
import json
import os
import statistics
import sys

ROOT = "/home/jli199/boptim_scratch/randrouter/opt_compare"
MARKS = (5, 10, 20, 30)


def curves(space):
    out = {}
    for d in sorted(glob.glob(f"{ROOT}/{space}/*_s*/")):
        name = os.path.basename(d.rstrip("/"))
        optim, seed = name.rsplit("_s", 1)
        p = d + "evaluations.jsonl"
        if not os.path.exists(p):
            continue
        vals = [json.loads(l)["value"] for l in open(p) if l.strip()]
        if not vals:
            continue
        best, cur = [], float("inf")
        for v in vals:
            cur = min(cur, v)
            best.append(cur)
        out.setdefault(optim, {})[int(seed)] = best
    return out


def report(space, log=print):
    cv = curves(space)
    res = {"space": space, "optimizers": {}}
    log(f"== {space}")
    for optim, runs in sorted(cv.items()):
        row = {"runs": len(runs)}
        cells = []
        for m in MARKS:
            xs = [b[m - 1] for b in runs.values() if len(b) >= m]
            if len(xs) >= 2:
                row[str(m)] = {"mean": statistics.mean(xs), "se": statistics.stdev(xs) / len(xs) ** 0.5, "n": len(xs)}
                cells.append(f"after {m}: {row[str(m)]['mean']:.2f} ± {row[str(m)]['se']:.2f} (n={len(xs)})")
            elif xs:
                row[str(m)] = {"mean": xs[0], "se": None, "n": 1}
                cells.append(f"after {m}: {xs[0]:.2f} (n=1)")
        res["optimizers"][optim] = row
        log(f"  {optim:<10} {len(runs)} runs; best value so far " + "; ".join(cells))
    for a, b in itertools.combinations(sorted(cv), 2):
        for m in (10, 30):
            fa = [r[m - 1] for r in cv[a].values() if len(r) >= m]
            fb = [r[m - 1] for r in cv[b].values() if len(r) >= m]
            if fa and fb:
                wins = sum(x < y for x in fa for y in fb)
                ties = sum(x == y for x in fa for y in fb)
                log(f"  {a} below {b} after {m} evaluations in {wins + 0.5 * ties:.0f} of {len(fa) * len(fb)} run pairs")
    return res


if __name__ == "__main__":
    spaces = sys.argv[1:] or sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(f"{ROOT}/*/"))
    allres = [report(s) for s in spaces]
    json.dump(allres, open(f"{ROOT}/report.json", "w"), indent=1)
