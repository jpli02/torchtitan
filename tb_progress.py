"""Progress of Terminal-Bench run dirs: python tb_progress.py ROOT name1 name2 ...
Per run: trials started, results.json written, resolved, failure modes."""
import collections
import glob
import json
import sys

root = sys.argv[1]
for name in sys.argv[2:]:
    trials = glob.glob(f"{root}/{name}/*/*/")
    res = glob.glob(f"{root}/{name}/*/*/results.json")
    ok = 0
    fm = collections.Counter()
    for f in res:
        try:
            d = json.load(open(f))
        except Exception:
            fm["unreadable"] += 1
            continue
        ok += bool(d.get("is_resolved"))
        fm[str(d.get("failure_mode") or ("RESOLVED" if d.get("is_resolved") else "other"))] += 1
    print(f"{name}: trials={len(trials)} results={len(res)} resolved={ok} modes={dict(fm)}")
