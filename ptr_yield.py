"""Usable yield of yoonholee trajectories once pointer-valued strings are
treated as missing.

Pointers ('$<hex>') don't resolve locally, so a trajectory containing one has
a hole: a missing observation (the model would learn to reason about output
it cannot see) or a missing assistant message/command (the supervised part).
Count, for terminus-2 / terminus-3-3 / mini-swe-agent rows with reward=1 on
tasks NOT in our 80:

  clean      no pointer anywhere              -> usable as-is
  obs_only   pointers only in obs slots        -> usable only if those obs can
                                                 be dropped/placeholdered
  tainted    pointer in msg or cmd             -> drop

Also: what fraction of pointer obs occur where the previous cmd was a
mark_task_complete (the confirm step), which is droppable without loss.
"""
import collections
import json
import os
import re

from datasets import load_dataset

PTR = re.compile(r"^\$[0-9a-f]+$")
TB = "/home/jli199/terminal_bench_eval/tb_tasks"
ours = set(d for d in os.listdir(TB) if os.path.isdir(os.path.join(TB, d)))
AGENTS = ("terminus-2", "terminus-3-3", "mini-swe-agent")

ds = load_dataset("yoonholee/terminalbench-trajectories", split="train", streaming=True)
cat = collections.Counter()
tasks_clean = collections.defaultdict(set)
lens_clean = collections.defaultdict(list)
n = 0
for r in ds:
    n += 1
    if r["agent"] not in AGENTS or r["task_name"] in ours or int(r.get("reward") or 0) != 1:
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
    msg_ptr = cmd_ptr = obs_ptr = 0
    for st in p:
        if isinstance(st.get("msg"), str) and PTR.match(st["msg"]):
            msg_ptr += 1
        if isinstance(st.get("obs"), str) and PTR.match(st["obs"]):
            obs_ptr += 1
        for t in (st.get("tools") or []):
            if isinstance(t.get("cmd"), str) and PTR.match(t["cmd"]):
                cmd_ptr += 1
    a = r["agent"]
    if msg_ptr or cmd_ptr:
        cat[(a, "tainted")] += 1
    elif obs_ptr:
        cat[(a, "obs_only")] += 1
    else:
        cat[(a, "clean")] += 1
        tasks_clean[a].add(r["task_name"])
        lens_clean[a].append(sum(1 for st in p if st.get("src") == "agent"))

print(f"scanned {n} rows")
for a in AGENTS:
    c, o, t = cat[(a, "clean")], cat[(a, "obs_only")], cat[(a, "tainted")]
    tot = c + o + t
    L = sorted(lens_clean[a])
    med = L[len(L) // 2] if L else 0
    print(f"{a:<16} reward=1 clean-task rows {tot:>5}: "
          f"CLEAN {c:>5} ({100 * c / max(tot, 1):4.0f}%) over {len(tasks_clean[a]):>2} tasks, "
          f"median {med} agent turns | obs_only {o:>5} | tainted {t:>5}")
