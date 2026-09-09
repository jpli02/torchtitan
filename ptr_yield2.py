"""Src-aware pointer accounting for yoonholee, plus a test of the encoding.

Two corrections to ptr_yield.py:
  1. A pointer in a USER 'msg' slot is the task description, recoverable from
     TB-2's task.yaml. It is not taint.
  2. Report obs-pointers separately from agent msg/cmd pointers, since only the
     latter touch supervised tokens.

Encoding test: if literal obs/msg strings never exceed some length while the
dataset says obs are truncated at 5,000 chars, then the pointers are exactly
the strings above that length -- hoisted by the page serializer and not
present in the scrape. That settles 'resolvable?' (no) and tells us what a
pointer obs was (a LONG output).
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
tasks = collections.defaultdict(set)
lit_len = collections.defaultdict(list)   # (agent, slot) -> literal lengths
n = 0
for r in ds:
    n += 1
    a = r["agent"]
    if a not in AGENTS or r["task_name"] in ours or int(r.get("reward") or 0) != 1:
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
    agent_msg = cmd = obs = user_msg = 0
    for st in p:
        src = st.get("src")
        m = st.get("msg")
        if isinstance(m, str):
            if PTR.match(m):
                if src == "agent":
                    agent_msg += 1
                else:
                    user_msg += 1
            else:
                lit_len[(a, "msg")].append(len(m))
        o = st.get("obs")
        if isinstance(o, str):
            if PTR.match(o):
                obs += 1
            else:
                lit_len[(a, "obs")].append(len(o))
        for t in (st.get("tools") or []):
            c = t.get("cmd")
            if isinstance(c, str):
                if PTR.match(c):
                    cmd += 1
                else:
                    lit_len[(a, "cmd")].append(len(c))
    if agent_msg or cmd:
        k = "agent_tainted"
    elif obs:
        k = "obs_only"
    else:
        k = "clean"
    cat[(a, k)] += 1
    if k != "agent_tainted":
        tasks[(a, k)].add(r["task_name"])
    if user_msg:
        cat[(a, "user_ptr_rows")] += 1

print(f"scanned {n}")
for a in AGENTS:
    c, o, t = cat[(a, "clean")], cat[(a, "obs_only")], cat[(a, "agent_tainted")]
    print(f"{a:<16} clean {c:>5} ({len(tasks[(a, 'clean')]):>2} tasks) | obs_only {o:>5} "
          f"({len(tasks[(a, 'obs_only')]):>2} tasks) | agent_tainted {t:>5} | rows with user-slot ptr {cat[(a, 'user_ptr_rows')]}")
print("\nliteral string lengths (max / p99 / p50):")
for (a, slot), L in sorted(lit_len.items()):
    L.sort()
    print(f"  {a:<16} {slot:<4} n={len(L):>6} max={L[-1]:>6} p99={L[int(.99 * len(L))]:>6} p50={L[len(L) // 2]:>5}")
