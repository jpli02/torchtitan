"""Per-task pass@2 on the 12-task set for every checkpoint evaluated on it.

Prints resolved/unresolved per task so the batching run can be read against the
measured noise floor rather than against a single aggregate number.
"""
import json, glob, os, collections

RUNS = ["clean_basethinking", "clean_continue10k", "clean_repro11k",
        "clean_longhorizon1k", "clean_router2k", "clean_batching5k"]

table = {}
for run in RUNS:
    per = collections.defaultdict(lambda: [0, 0])  # task -> [solved, trials]
    for f in glob.glob(f"/tmp/tb_runs/{run}/*/*/results.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        # results.json shape varies by tb version; probe the common keys
        tid = d.get("task_id") or d.get("task_name") or os.path.basename(
            os.path.dirname(os.path.dirname(f)))
        ok = d.get("is_resolved")
        if ok is None:
            ok = (d.get("resolved") is True) or (d.get("status") == "resolved")
        per[tid][1] += 1
        per[tid][0] += 1 if ok else 0
    if per:
        table[run] = per

tasks = sorted({t for p in table.values() for t in p})
if not tasks:
    print("no results parsed")
else:
    w = max(len(t) for t in tasks) + 1
    hdr = "task".ljust(w) + "".join(r.replace("clean_", "")[:13].rjust(15) for r in table)
    print(hdr); print("-" * len(hdr))
    for t in tasks:
        row = t.ljust(w)
        for r in table:
            s, n = table[r].get(t, [0, 0])
            row += (f"{s}/{n}" if n else "-").rjust(15)
        print(row)
    print("-" * len(hdr))
    tot = "TOTAL solved".ljust(w)
    for r in table:
        s = sum(v[0] for v in table[r].values())
        n = sum(v[1] for v in table[r].values())
        tot += f"{s}/{n}".rjust(15)
    print(tot)
    tot2 = "tasks >=1 pass".ljust(w)
    for r in table:
        k = sum(1 for v in table[r].values() if v[0] > 0)
        tot2 += f"{k}/{len(table[r])}".rjust(15)
    print(tot2)
