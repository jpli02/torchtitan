"""Ground truth: how many commands per turn did each checkpoint actually emit
during its 12-task eval, and did any task get zero usable responses?

The probe is a proxy measured on one synthetic prefix. This is the real thing:
every assistant turn the model produced against live terminal state.
"""
import json, glob, re, collections, os

def extract(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

for run in ["clean_continue10k", "clean_batching5k"]:
    hist = collections.Counter()
    turns = bad = 0
    per_task = collections.defaultdict(lambda: [0, 0])  # task -> [usable, total]
    for f in glob.glob(f"/tmp/tb_runs/{run}/*/*/agent-logs/episode-*/debug.json"):
        task = f.split(f"{run}/")[1].split("/")[1] if f"{run}/" in f else "?"
        per_task[task][1] += 1
        try:
            j = json.loads(json.load(open(f))["original_response"])
            msg = j["choices"][0]["message"].get("content")
            txt = msg if isinstance(msg, str) else "".join(
                b.get("text", "") for b in (msg or []))
            d = extract(txt or "")
        except Exception:
            d = None
        if d is None:
            bad += 1
            continue
        per_task[task][0] += 1
        c = d.get("commands")
        if isinstance(c, list):
            turns += 1
            hist[min(len(c), 6)] += 1
    tot = sum(hist.values()) or 1
    mean = sum(k * v for k, v in hist.items()) / tot
    print(f"=== {run} ===")
    print(f"  parsed turns: {turns}   unparseable: {bad}")
    print(f"  mean commands/turn: {mean:.3f}   max bucket: {max(hist) if hist else 0}")
    print(f"  histogram (cmds -> turns): {dict(sorted(hist.items()))}")
    dead = [t for t, (u, n) in per_task.items() if n and u == 0]
    print(f"  tasks with ZERO usable responses: {dead if dead else 'none'}")
    print()
