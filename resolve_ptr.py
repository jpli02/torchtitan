"""Test hypotheses for the '$<hex>' pointers in yoonholee steps JSON.

Hypothesis: strings are interned in document order; '$<hex>' is the index of
an earlier distinct string. Variants differ in WHAT gets counted (all string
values, only non-pointer values, only strings above a length, keys too) and
whether repeats get their own index. A variant is judged by type-consistency:
a pointer sitting in an 'obs' slot should resolve to something that looks like
terminal output; one in 'cmd' to a command; one in a user 'msg' to a task
statement. Report per-variant consistency over many rows.
"""
import json
import re
import sys

from datasets import load_dataset

PTR = re.compile(r"^\$[0-9a-f]+$")


def walk(obj, out, opts):
    """Collect string values in document order per the variant's rules."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if opts["keys"]:
                out.append(k)
            walk(v, out, opts)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, out, opts)
    elif isinstance(obj, str):
        if PTR.match(obj) and not opts["count_ptr"]:
            return
        if len(obj) < opts["minlen"]:
            return
        if opts["distinct"]:
            if obj not in opts["_seen"]:
                opts["_seen"].add(obj)
                out.append(obj)
        else:
            out.append(obj)


def looks_ok(slot, s):
    if not isinstance(s, str):
        return False
    if slot == "obs":
        return ("Terminal Output" in s or "root@" in s or "<returncode>" in s
                or "Current terminal state" in s or "Previous response" in s)
    if slot == "cmd":
        return "\n" in s or len(s) < 400
    if slot == "msg":
        return len(s) > 20
    return True


VARIANTS = {
    "all_strings_ordered":        dict(keys=False, count_ptr=False, minlen=0, distinct=False),
    "distinct_strings":           dict(keys=False, count_ptr=False, minlen=0, distinct=True),
    "distinct_incl_ptr":          dict(keys=False, count_ptr=True,  minlen=0, distinct=True),
    "distinct_keys_too":          dict(keys=True,  count_ptr=False, minlen=0, distinct=True),
    "all_incl_keys":              dict(keys=True,  count_ptr=False, minlen=0, distinct=False),
    "distinct_len>=8":            dict(keys=False, count_ptr=False, minlen=8, distinct=True),
}

ds = load_dataset("yoonholee/terminalbench-trajectories", split="train", streaming=True)
rows = []
for r in ds:
    s = r.get("steps")
    if s in (None, "null", ""):
        continue
    if "$" not in s:
        continue
    try:
        p = json.loads(s)
    except Exception:
        continue
    if isinstance(p, list) and any(isinstance(st.get("obs"), str) and PTR.match(st["obs"]) for st in p):
        rows.append((r["agent"], p))
    if len(rows) >= 40:
        break
print(f"test rows with obs pointers: {len(rows)}")

for name, base in VARIANTS.items():
    ok = tot = 0
    example = None
    for agent, p in rows:
        opts = dict(base); opts["_seen"] = set()
        table = []
        walk(p, table, opts)
        for st in p:
            for slot in ("msg", "obs"):
                v = st.get(slot)
                if isinstance(v, str) and PTR.match(v):
                    idx = int(v[1:], 16)
                    tot += 1
                    res = table[idx] if idx < len(table) else None
                    if looks_ok(slot, res):
                        ok += 1
                    elif example is None:
                        example = (slot, v, (res or "<OUT OF RANGE>")[:90])
            for t in (st.get("tools") or []):
                c = t.get("cmd")
                if isinstance(c, str) and PTR.match(c):
                    idx = int(c[1:], 16)
                    tot += 1
                    res = table[idx] if idx < len(table) else None
                    if looks_ok("cmd", res):
                        ok += 1
                    elif example is None:
                        example = ("cmd", c, (res or "<OUT OF RANGE>")[:90])
    print(f"{name:<24} consistent {ok:>5}/{tot:<5} ({100 * ok / max(tot, 1):5.1f}%)  "
          f"first-miss={example}")
