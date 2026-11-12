"""Terminal-Bench counts for a run name merged over its main root and its resume root(s):
  python tb_counts_merged.py ROOT NAME                      -> "NAME: results=N resolved=K timeouts=T (main a, resume b)"
  python tb_counts_merged.py ROOT NAME --remaining t1 t2 .. -> the tasks of the list with no results.json in either root
A task that has a result in both roots counts once (the first found wins)."""
import glob
import json
import sys

root, name = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
seen = {}
per_root = {}
for label, pattern in (("main", f"{root}/{name}/*/*/results.json"), ("resume", f"{root}/resume_{name}/{name}/*/*/results.json")):
    n = 0
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        t = d.get("task_id")
        if t in seen:
            continue
        seen[t] = d
        n += 1
    per_root[label] = n
if rest and rest[0] == "--remaining":
    print(" ".join(t for t in rest[1:] if t not in seen))
else:
    ok = sum(1 for d in seen.values() if d.get("is_resolved"))
    to = sum(1 for d in seen.values() if d.get("failure_mode") == "agent_timeout")
    print(f"{name}: results={len(seen)} resolved={ok} timeouts={to} (main {per_root['main']}, resume {per_root['resume']})")
