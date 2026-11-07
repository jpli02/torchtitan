"""Decompose Terminal-Bench trials: python tb_failure_decomp.py ROOT run1 [run2 ...]
Per trial: outcome (resolved / claimed-done-but-wrong / timeout / other), fraction of parser tests passed (partial
credit), episodes, whether the agent ever declared task_complete, and the wall time. Per run: counts and mean partial
credit per outcome, plus the share of unresolved trials with >0 partial credit (near-misses)."""
import collections
import glob
import json
import os
import statistics
import sys

root = sys.argv[1]
for run in sys.argv[2:]:
    rows = []
    for tdir in sorted(glob.glob(f"{root}/{run}/*/*/")):
        try:
            res = json.load(open(f"{tdir}/results.json"))
        except Exception:
            continue
        pr = res.get("parser_results") or {}
        frac = (sum(1 for v in pr.values() if v == "passed") / len(pr)) if pr else 0.0
        eps = sorted(glob.glob(f"{tdir}/agent-logs/episode-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
        claimed = False
        for e in eps[-3:]:
            try:
                if '"task_complete": true' in open(f"{e}/response.txt", errors="replace").read().replace(" ", "").replace('"task_complete":true', '"task_complete": true'):
                    claimed = True
            except OSError:
                pass
        if res.get("is_resolved"):
            cat = "resolved"
        elif res.get("failure_mode") == "agent_timeout":
            cat = "timeout"
        elif claimed:
            cat = "claimed_done_wrong"
        else:
            cat = "stopped_wrong"
        rows.append({"task": res.get("task_id"), "cat": cat, "frac": frac, "n_tests": len(pr), "episodes": len(eps), "claimed": claimed})
    if not rows:
        print(f"{run}: no trials")
        continue
    cats = collections.Counter(r["cat"] for r in rows)
    print(f"\n== {run}: {len(rows)} trials; {dict(cats)}")
    for cat in ("resolved", "claimed_done_wrong", "stopped_wrong", "timeout"):
        sel = [r for r in rows if r["cat"] == cat]
        if sel:
            print(f"   {cat:18s} n={len(sel):2d}  partial credit mean {statistics.mean(r['frac'] for r in sel):.2f}  "
                  f">0: {sum(r['frac'] > 0 for r in sel)}/{len(sel)}  episodes mean {statistics.mean(r['episodes'] for r in sel):.1f}")
    unres = [r for r in rows if r["cat"] != "resolved"]
    if unres:
        print(f"   unresolved with some tests passing: {sum(r['frac'] > 0 for r in unres)}/{len(unres)}; "
              f"all-or-nothing tasks (1 test): {sum(r['n_tests'] == 1 for r in rows)}/{len(rows)}")
    if "--tasks" in sys.argv:
        for r in sorted(rows, key=lambda r: (r["cat"], r["task"])):
            print(f"   {r['task'][:44]:44s} {r['cat']:18s} credit {r['frac']:.2f} ({r['n_tests']} tests) ep={r['episodes']:3d} claimed={r['claimed']}")
