"""How much of the 420s budget do the early-quitting trials leave unused?

batching5k converts timeouts into early completions: 14 timeouts / 10 completed
vs continue10k's 17 / 6. All of its non-timeout unresolved trials declare
task_complete=true at ~28% of tests passing. If those trials also stop well
short of the clock, then gating task_complete buys real extra work for free --
and the size of that gap is the upside.
"""
import json, glob, datetime, statistics

def secs(a, b):
    try:
        pa = datetime.datetime.fromisoformat(a.replace("Z", "+00:00"))
        pb = datetime.datetime.fromisoformat(b.replace("Z", "+00:00"))
        return (pb - pa).total_seconds()
    except Exception:
        return None

for run in ["clean_continue10k", "clean_batching5k"]:
    by_mode = {}
    for f in glob.glob(f"/tmp/tb_runs/{run}/*/*/results.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        s = secs(d.get("agent_started_at"), d.get("agent_ended_at"))
        if s is None:
            continue
        mode = "RESOLVED" if d.get("is_resolved") else (d.get("failure_mode") or "none")
        by_mode.setdefault(mode, []).append(s)
    print(f"=== {run} ===")
    for mode, xs in sorted(by_mode.items(), key=lambda kv: -len(kv[1])):
        med = statistics.median(xs)
        print(f"  {mode:<22} n={len(xs):<3} median {med:6.1f}s   "
              f"unused of 420s: {420-med:6.1f}s ({100*(420-med)/420:.0f}%)")
    print()
