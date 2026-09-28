"""Where does the tb80sft prompt/trajectory mismatch come from?

The rendered TB-2 rows have median task-vs-trajectory keyword overlap 0.08
(68% under 0.10) against a TerminalTraj control of 0.59 (0% under 0.10). Either
the yoonholee source mislabels task_name, or build_tb80_sft.py paired them
wrong. Test at the SOURCE, before my code touches anything:

  agent-vs-META[task_name]  overlap between the row's agent messages and the
                            instruction of the task it is labelled with
  agent-vs-ROW-OWN-user     for rows whose user msg is a real string (not a
                            pointer), overlap with that in-row instruction
  prefix-match              does the in-row user msg equal META[task_name]?

If agent-vs-META is high at the source, my rendering is the bug. If it is low
at the source too, task_name is wrong in the data.
"""
import json
import os
import re

from datasets import load_dataset

META = json.load(open("/home/jli199/boptim_scratch/tb2_meta.json"))
TB = "/home/jli199/terminal_bench_eval/tb_tasks"
OURS = set(d for d in os.listdir(TB) if os.path.isdir(os.path.join(TB, d)))
PTR = re.compile(r"^\$[0-9a-f]+$")
STOP = set("the a an and or to of in for on with is are be by as at from this that "
           "it your you will should must can each all any into use using file files task".split())


def kw(s):
    return set(w for w in re.findall(r"[a-z][a-z0-9_.-]{3,}", (s or "").lower()) if w not in STOP)


def is_ptr(v):
    return isinstance(v, str) and PTR.match(v) is not None


ds = load_dataset("yoonholee/terminalbench-trajectories", split="train", streaming=True)
n = real_user = user_matches_meta = shown = 0
ov_meta, ov_own = [], []
for r in ds:
    t = r["task_name"]
    if t in OURS or r["agent"] != "terminus-2" or int(r.get("reward") or 0) != 1:
        continue
    s = r.get("steps")
    if s in (None, "null", ""):
        continue
    try:
        p = json.loads(s)
    except Exception:
        continue
    if not isinstance(p, list):
        continue
    n += 1
    agent_txt = " ".join(str(x.get("msg") or "") for x in p
                         if x.get("src") == "agent" and not is_ptr(x.get("msg")))[:6000]
    inst = META.get(t, {}).get("instruction", "")
    ov_meta.append(len(kw(inst) & kw(agent_txt)) / max(len(kw(inst)), 1))
    um = [str(x.get("msg")) for x in p
          if x.get("src") == "user" and len(str(x.get("msg") or "")) > 40 and not is_ptr(x.get("msg"))]
    if um:
        real_user += 1
        if inst[:60] and inst[:60] in um[0]:
            user_matches_meta += 1
        ov_own.append(len(kw(um[0]) & kw(agent_txt)) / max(len(kw(um[0])), 1))
        if shown < 3:
            shown += 1
            print(f"--- task_name={t}\n   META inst: {inst[:90]}\n   row user : {um[0][:90]}\n   agent[0] : {agent_txt[:90]}")
    if n >= 1500:
        break

ov_meta.sort(); ov_own.sort()
print(f"\nSOURCE rows checked {n}   rows with a REAL (non-pointer) user msg {real_user}")
print(f"  agent-vs-META[task_name] overlap: median {ov_meta[len(ov_meta)//2]:.2f}   "
      f"<0.10: {sum(1 for x in ov_meta if x < 0.10)}/{len(ov_meta)}")
if ov_own:
    print(f"  agent-vs-ROW-OWN-user overlap:   median {ov_own[len(ov_own)//2]:.2f}   "
          f"<0.10: {sum(1 for x in ov_own if x < 0.10)}/{len(ov_own)}")
print(f"  real user msg prefix-matches META[task_name]: {user_matches_meta}/{real_user}")
