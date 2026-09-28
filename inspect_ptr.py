"""Decode the string-interning scheme in yoonholee/terminalbench-trajectories.

Some step values read like '$34' / '$3f': the 'steps' JSON appears to replace
repeated strings with pointers. Unresolved pointers would silently corrupt
converted rows, so before building anything: how are they encoded, how often
do they occur per agent and field, and is there a string table to resolve
against.
"""
import collections
import json
import re

from datasets import load_dataset

PTR = re.compile(r"\$[0-9a-f]+")
ds = load_dataset("yoonholee/terminalbench-trajectories", split="train", streaming=True)
n = 0
ptr = collections.Counter()
tot = collections.Counter()
shown_raw = shown_ctx = False
for r in ds:
    s = r.get("steps")
    if s in (None, "null", ""):
        continue
    n += 1
    if not shown_raw:
        print("RAW head:", s[:300].replace("\n", " "))
        print("RAW tail:", s[-300:].replace("\n", " "))
        shown_raw = True
    try:
        p = json.loads(s)
    except Exception:
        continue
    if isinstance(p, dict):
        print("TOP-LEVEL IS DICT, keys:", list(p.keys())[:10])
        break
    for st in p:
        for k in ("msg", "obs"):
            v = st.get(k)
            if isinstance(v, str):
                tot[(r["agent"], k)] += 1
                if PTR.fullmatch(v):
                    ptr[(r["agent"], k)] += 1
        for t in (st.get("tools") or []):
            c = t.get("cmd")
            if isinstance(c, str):
                tot[(r["agent"], "cmd")] += 1
                if PTR.fullmatch(c):
                    ptr[(r["agent"], "cmd")] += 1
    if not shown_ctx:
        m = re.search(r'"\$[0-9a-f]+"', s)
        if m:
            i = m.start()
            print("POINTER CONTEXT:", s[max(0, i - 260):i + 80].replace("\n", " "))
            shown_ctx = True
    if n >= 3000:
        break
print(f"rows scanned {n}")
for (a, k), c in sorted(tot.items()):
    if a in ("terminus-2", "terminus-3-3", "mini-swe-agent", "openhands", "claude-code"):
        print(f"  {a:<16} {k:<4} pointer-valued {ptr[(a, k)]:>6}/{c:<6} ({100 * ptr[(a, k)] / c:.1f}%)")
