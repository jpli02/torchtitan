"""Do the completed-but-failing trials declare victory prematurely?

batching5k turned 4 timeouts into completions, but those completions still fail
their tests. If the agent is setting task_complete=true while the task's own
tests would fail, that is a targeted, teachable error -- verify before claiming
done -- rather than a capability ceiling.
"""
import json, glob, re, collections

def extract(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

for run in ["clean_continue10k", "clean_batching5k"]:
    claimed = quiet = 0
    part = []
    for rf in glob.glob(f"/tmp/tb_runs/{run}/*/*/results.json"):
        try:
            d = json.load(open(rf))
        except Exception:
            continue
        if d.get("is_resolved") or d.get("failure_mode") == "agent_timeout":
            continue
        pr = d.get("parser_results") or {}
        n = len(pr) or 1
        p = sum(1 for v in pr.values() if str(v).lower() in ("passed", "true", "ok"))
        part.append(p / n)
        # did any turn in this trial claim completion?
        trial = rf.rsplit("/", 1)[0]
        said = False
        for f in glob.glob(f"{trial}/agent-logs/episode-*/debug.json"):
            try:
                j = json.loads(json.load(open(f))["original_response"])
                msg = j["choices"][0]["message"].get("content")
                txt = msg if isinstance(msg, str) else "".join(
                    b.get("text", "") for b in (msg or []))
                o = extract(txt)
                if o and o.get("task_complete") is True:
                    said = True
                    break
            except Exception:
                pass
        claimed += said
        quiet += (not said)
    tot = claimed + quiet
    if not tot:
        continue
    avg = sum(part) / len(part) if part else 0
    print(f"=== {run} ===")
    print(f"  non-timeout, unresolved trials: {tot}")
    print(f"  declared task_complete=true anyway: {claimed}   never declared: {quiet}")
    print(f"  mean fraction of tests passed in those trials: {avg:.1%}")
    print()
