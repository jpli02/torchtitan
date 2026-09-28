"""Why do 62-65 of 80 tasks hit agent_timeout?

Candidate mechanisms, which imply completely different fixes:
  A. too few turns fit in 420s  -> latency problem (tokens/turn, prefill growth)
  B. plenty of turns, no progress -> competence problem
  C. repetition loops            -> the model re-issues the same action
Measure which it is on the cleanest 80-task run (continue10k, 0.9% failure).
"""
import json, glob, collections, statistics, hashlib

RUN = "e80_c10k80"
per_task_turns, per_task_mode = {}, {}
for f in glob.glob(f"/tmp/tb_runs/{RUN}_s*/*/*/results.json"):
    d = json.load(open(f))
    per_task_mode[d["task_id"]] = d.get("failure_mode") or ("SOLVED" if d.get("is_resolved") else "?")

toks, turns, dup_frac = {}, {}, {}
for f in glob.glob(f"/tmp/tb_runs/{RUN}_s*/*/*/agent-logs/episode-*/debug.json"):
    task = f.split("/tmp/tb_runs/")[1].split("/")[1]
    try:
        d = json.load(open(f)); r = d.get("original_response")
        j = json.loads(r) if isinstance(r, str) and r.strip().startswith("{") else None
    except Exception:
        continue
    if not j:
        continue
    u = j.get("usage") or {}
    turns[task] = turns.get(task, 0) + 1
    toks.setdefault(task, []).append(u.get("completion_tokens") or 0)
    txt = "".join(b.get("text", "") for b in j["choices"][0]["message"].get("content", [])) \
        if isinstance(j["choices"][0]["message"].get("content"), list) \
        else j["choices"][0]["message"].get("content") or ""
    dup_frac.setdefault(task, []).append(hashlib.sha1(txt.strip().encode()).hexdigest()[:12])

groups = collections.defaultdict(list)
for t, m in per_task_mode.items():
    groups[m].append(t)

print(f"{'mode':22s} {'n':>3} {'med turns':>10} {'med tok/turn':>13} {'med %dup resp':>14}")
for mode, tasks in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    tn, tk, dp = [], [], []
    for t in tasks:
        if t in turns:
            tn.append(turns[t])
            tk.append(statistics.median(toks[t]))
            h = dup_frac[t]
            dp.append(100 * (1 - len(set(h)) / len(h)) if h else 0)
    if tn:
        print(f"{mode:22s} {len(tasks):>3} {statistics.median(tn):>10.0f} "
              f"{statistics.median(tk):>13.0f} {statistics.median(dp):>13.0f}%")

to = [t for t, m in per_task_mode.items() if m == "agent_timeout" and t in turns]
if to:
    tt = sorted(turns[t] for t in to)
    print(f"\nagent_timeout turn counts: min={tt[0]} p25={tt[len(tt)//4]} "
          f"median={tt[len(tt)//2]} p75={tt[3*len(tt)//4]} max={tt[-1]}")
    allt = [x for t in to for x in toks[t]]
    print(f"agent_timeout tokens/turn: median={statistics.median(allt):.0f} "
          f"mean={statistics.mean(allt):.0f} max={max(allt)}")
