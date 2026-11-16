#!/usr/bin/env python
"""Paired, per-item comparison of routers on one BFCL item set (same items, same checker).

For every corrected (calibrated-fill) router on the set and every plain reference (full depth, released gate points) it
reports: items solved by both / only the router / only the reference, the exact two-sided sign test on the discordant
items, and a paired bootstrap 95 % interval of the accuracy difference (router - reference, percentage points).

  python boptim_lcb/bfcl_paired.py heldout        # or: search, live
Writes <work>/paired_<set>.json and prints a table.
"""
import json
import math
import os
import random
import sys

WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
FILL = WORK + "_fill"
REFS = [(1.0, 1.0, 1.0), (0.3, 0.3, 0.3), (0.25, 0.25, 0.25), (0.2, 0.2, 0.2)]
REF_NAMES = {(1.0, 1.0, 1.0): "full depth", (0.3, 0.3, 0.3): "plain gate 0.3", (0.25, 0.25, 0.25): "plain gate 0.25",
             (0.2, 0.2, 0.2): "plain gate 0.2"}


def _rows(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        if e.get("status") == "ok" and e.get("run_dir") and os.path.exists(os.path.join(e["run_dir"], "graded.json")):
            e["key"] = tuple(round(float(v), 4) for v in e["thresholds"][:3])
            out.append(e)
    return out


def _items(e):
    g = json.load(open(os.path.join(e["run_dir"], "graded.json")))
    return {d["id"]: bool(d["ok"]) for d in g["details"]}


def _sign_p(a, b):
    """Exact two-sided sign test for a vs b discordant counts."""
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def _boot(diffs, reps=10000, seed=7):
    """Paired bootstrap of the mean difference. The per-item differences are -1 / 0 / +1, so one resample of n items is a
    multinomial draw of the three counts (same distribution as resampling the items, and fast)."""
    n = len(diffs)
    up, down = sum(1 for d in diffs if d > 0), sum(1 for d in diffs if d < 0)
    try:
        import numpy as np
        c = np.random.default_rng(seed).multinomial(n, [up / n, down / n, (n - up - down) / n], size=reps)
        stats = np.sort(100.0 * (c[:, 0] - c[:, 1]) / n)
        return float(stats[int(0.025 * reps)]), float(stats[int(0.975 * reps) - 1])
    except ImportError:
        rng = random.Random(seed)
        stats = sorted(100.0 * sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
        return stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


def compute(name, write=True):
    """Rows of the paired comparison for one item set ("search", "heldout", "live"), most loops first."""
    if name == "search":
        plain = [e for e in _rows(f"{WORK}/evaluations.jsonl") if not e.get("extra_cfg")]
        corr = [e for e in _rows(f"{FILL}/evaluations.jsonl") + _rows(f"{WORK}/lab_search/evaluations.jsonl")]
    else:
        plain = [e for e in _rows(f"{WORK}/{name}/evaluations.jsonl") + _rows(f"{WORK}/lab_{name}/evaluations.jsonl") if not e.get("extra_cfg")]
        corr = _rows(f"{WORK}/lab_{name}/evaluations.jsonl")
    corr = [e for e in corr if (e.get("extra_cfg") or {}).get("decode_fill_maps") and not (e.get("extra_cfg") or {}).get("decode_exit_conf")]
    refs = {}
    for e in plain:
        if e["key"] in REFS and e["key"] not in refs:
            refs[e["key"]] = e
    seen, out = set(), []
    for e in corr:
        if e["key"] in seen:
            continue
        seen.add(e["key"])
        a = _items(e)
        row = {"thresholds": list(e["key"]), "acc_pct": e["acc_pct"], "exec_loops": e["exec_loops"], "n": len(a), "vs": {}}
        for k in REFS:
            if k not in refs:
                continue
            b = _items(refs[k])
            ids = sorted(set(a) & set(b))
            only_a = sum(1 for i in ids if a[i] and not b[i])
            only_b = sum(1 for i in ids if b[i] and not a[i])
            both = sum(1 for i in ids if a[i] and b[i])
            lo, hi = _boot([int(a[i]) - int(b[i]) for i in ids])
            row["vs"][REF_NAMES[k]] = {"n": len(ids), "both": both, "only_router": only_a, "only_ref": only_b,
                                       "diff_pts": round(100.0 * (only_a - only_b) / len(ids), 2), "ci95": [round(lo, 2), round(hi, 2)],
                                       "sign_p": round(_sign_p(only_a, only_b), 4), "ref_acc_pct": refs[k]["acc_pct"], "ref_loops": refs[k]["exec_loops"]}
        out.append(row)
    out.sort(key=lambda r: -r["exec_loops"])
    if write:
        json.dump({"set": name, "rows": out}, open(f"{WORK}/paired_{name}.json", "w"), indent=1)
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "heldout"
    for r in compute(name):
        print("router (%.4f, %.4f, %.4f) + fill: %.2f %% at %.3f loops" % (*r["thresholds"], r["acc_pct"], r["exec_loops"]))
        for nm, v in r["vs"].items():
            print("   vs %-16s (%.2f %% at %.3f): only router %3d, only reference %3d, diff %+.2f pts, 95 %% CI [%+.2f, %+.2f], sign p %.3f"
                  % (nm, v["ref_acc_pct"], v["ref_loops"], v["only_router"], v["only_ref"], v["diff_pts"], v["ci95"][0], v["ci95"][1], v["sign_p"]))


if __name__ == "__main__":
    main()
