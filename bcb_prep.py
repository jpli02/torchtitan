"""Prepare BigCodeBench (v0.1.4) as JSONL for depth_eval --task bcb.

Writes TMP/bcb_hard.jsonl (all 148 Hard tasks) and TMP/bcb_full300.jsonl (300 tasks sampled from
the full 1,140 with the Hard tasks excluded, stratified over the first listed library, seed 0).
Row: {task_id, instruct_prompt, code_prompt, entry_point, test, libs}
"""
import collections
import json
import random

from datasets import load_dataset

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"


import ast


def libs_of(v):
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except Exception:
            v = [v]
    return [str(x) for x in v]


def rows_of(ds):
    return [{"task_id": r["task_id"], "instruct_prompt": r["instruct_prompt"], "code_prompt": r["code_prompt"],
             "entry_point": r["entry_point"], "test": r["test"], "libs": libs_of(r["libs"])} for r in ds]


hard = rows_of(load_dataset("bigcode/bigcodebench-hard", split="v0.1.4"))
with open(f"{TMP}/bcb_hard.jsonl", "w") as f:
    for r in hard:
        f.write(json.dumps(r) + "\n")
hard_ids = {r["task_id"] for r in hard}
full = [r for r in rows_of(load_dataset("bigcode/bigcodebench", split="v0.1.4")) if r["task_id"] not in hard_ids]
random.seed(0)
groups = collections.defaultdict(list)
for r in full:
    groups[r["libs"][0] if r["libs"] else "none"].append(r)
per = max(1, 300 // len(groups))
sub = []
for g in sorted(groups):
    sub += random.sample(groups[g], min(per, len(groups[g])))
rest = [r for r in full if r not in sub]
random.shuffle(rest)
sub += rest[: max(0, 300 - len(sub))]
sub = sub[:300]
with open(f"{TMP}/bcb_full300.jsonl", "w") as f:
    for r in sub:
        f.write(json.dumps(r) + "\n")
print(f"hard: {len(hard)} tasks; full300: {len(sub)} tasks over {len(groups)} lead libraries; "
      f"test suites median {sorted(len(r['test']) for r in sub)[len(sub)//2]} chars")
