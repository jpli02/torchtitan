"""Composition of the take-turns distillation sets: rows and distinct tasks (matched to the pool task.yaml instructions).
Usage: python dataset_tasks.py"""
import collections
import glob
import json
import os
import re

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
POOL = "/home/jli199/boptim_scratch/tb_variants/pool"


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


instr = {}
for ty in glob.glob(f"{POOL}/*/task.yaml"):
    txt = open(ty, errors="replace").read()
    m = re.search(r"instruction:\s*\|?-?\s*\n?(.*?)(?:\n[a-z_]+:|\Z)", txt, re.S)
    if m:
        instr[os.path.basename(os.path.dirname(ty))] = norm(m.group(1))[:200]
pool_tasks = sorted(instr)
base_of = lambda t: re.sub(r"__.*$", "", t)  # noqa: E731  variant id -> base task name


def task_of(row):
    first = norm(next((m["content"] for m in row["messages"] if m["role"] == "user"), ""))
    best, best_len = None, 0
    for t, key in instr.items():
        if key and key in first and len(key) > best_len:
            best, best_len = t, len(key)
    return best


def summarize(name, paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    tasks = [task_of(r) for r in rows]
    c = collections.Counter(t for t in tasks if t)
    bases = {base_of(t) for t in c}
    turns = [sum(1 for m in r["messages"] if m["role"] == "assistant") for r in rows]
    print(f"{name}: rows={len(rows)} unmatched={sum(1 for t in tasks if t is None)} distinct variant tasks={len(c)} "
          f"(base tasks {len(bases)}); rows/task: 1x {sum(1 for v in c.values() if v == 1)}, 2x {sum(1 for v in c.values() if v == 2)}, "
          f">=3x {sum(1 for v in c.values() if v >= 3)}; assistant turns/row mean {sum(turns)/len(turns):.1f}")
    return c


print(f"pool: {len(pool_tasks)} variant tasks over {len({base_of(t) for t in pool_tasks})} base tasks")
r1 = summarize("take-turns r1 (training rows)", [f"{TMP}/tvary_r1_new.jsonl"])
r2 = summarize("take-turns r2 (shards)", sorted(glob.glob(f"{TMP}/tvary_r2_sh*.jsonl")))
mixed12 = [json.loads(l) for l in open(f"{TMP}/tvary_r12_mixed.jsonl") if l.strip()]
v3keys = set()
for l in open("/home/jli199/boptim_scratch/tb80sft/train_v3.jsonl"):
    if l.strip():
        v3keys.add(json.dumps(json.loads(l)["messages"], sort_keys=True)[:4000])
new12 = [r for r in mixed12 if json.dumps(r["messages"], sort_keys=True)[:4000] not in v3keys]
c12 = collections.Counter(t for t in (task_of(r) for r in new12) if t)
print(f"take-turns r1+r2 training rows: {len(new12)} new rows over {len(c12)} variant tasks ({len({base_of(t) for t in c12})} base tasks); union r1|r2 = {len(set(r1) | set(r2))} tasks")
lists = {f: len(open(f"{TMP}/{f}").read().split()) for f in ("train_tasks_r1.txt", "train_tasks_tvary.txt") if os.path.exists(f"{TMP}/{f}")}
print("task lists used for the train-task evals:", lists)
print(f"generation yield r1: {len(r1)}/{len(pool_tasks)} tasks with a resolved trajectory = {100*len(r1)/len(pool_tasks):.0f} %; r1|r2: {len(set(r1)|set(r2))}/{len(pool_tasks)} = {100*len(set(r1)|set(r2))/len(pool_tasks):.0f} %")
