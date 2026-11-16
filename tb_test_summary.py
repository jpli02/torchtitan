"""One-line summary of a Terminal-Bench run directory under /tmp/tb_runs (python 3.6 compatible).

  python3 -I tb_test_summary.py <run name> [root]
"""
import collections
import glob
import json
import sys

name = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else "/tmp/tb_runs"
per, fm = collections.defaultdict(lambda: [0, 0]), collections.Counter()
for f in glob.glob("%s/%s/*/*/results.json" % (root, name)):
    d = json.load(open(f))
    per[d.get("task_id")][1] += 1
    per[d.get("task_id")][0] += 1 if d.get("is_resolved") else 0
    fm[str(d.get("failure_mode"))] += 1
print("%s: %d/%d solved, tasks with a solve %d/%d, failure modes %s" % (
    name, sum(v[0] for v in per.values()), sum(v[1] for v in per.values()), sum(1 for v in per.values() if v[0]), len(per), dict(fm)))
print("   " + ", ".join("%s %d/%d" % (t, s, c) for t, (s, c) in sorted(per.items())))
