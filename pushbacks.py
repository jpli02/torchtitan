"""How much self-distillation data did the gate actually generate?

The gate proves a counterfactual: on csv-to-parquet the model quit at episode 19
with wrong data, was refused, and fixed it by episode 29. That is a training
signal we own -- (premature claim -> pushback -> repair -> success) trajectories
show the model its own correct behaviour.

The question is whether there are enough of them to SFT on. Count pushback
events, and how many of the trials containing one went on to resolve.
"""
import json, glob, re, collections

def obj(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

for run in ["gated_batching5k", "gated_continue10k",
            "sweep_step1000", "sweep_step2000"]:
    trials = pushed = pushed_resolved = 0
    events = 0
    for rf in glob.glob(f"/tmp/tb_runs/{run}/*/*/results.json"):
        try:
            d = json.load(open(rf))
        except Exception:
            continue
        trials += 1
        trial = rf.rsplit("/", 1)[0]
        n_here = 0
        eps = sorted(glob.glob(f"{trial}/agent-logs/episode-*/debug.json"))
        for f in eps:
            try:
                j = json.loads(json.load(open(f))["original_response"])
                msg = j["choices"][0]["message"].get("content")
                txt = msg if isinstance(msg, str) else "".join(
                    b.get("text", "") for b in (msg or []))
                o = obj(txt)
            except Exception:
                continue
            # the shape the gate refuses: claims done, ran nothing
            if o and o.get("task_complete") is True and not (o.get("commands") or []):
                n_here += 1
        if n_here:
            pushed += 1
            events += n_here
            if d.get("is_resolved"):
                pushed_resolved += 1
    if not trials:
        continue
    print(f"=== {run} ===")
    print(f"  trials {trials}   trials with >=1 empty-handed completion claim: {pushed}")
    print(f"  total such claims: {events}   of those trials, resolved: {pushed_resolved}")
    print()
