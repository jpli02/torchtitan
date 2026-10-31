"""Prepare multiple-choice benchmarks as JSONL for depth_eval --task mc.

Writes TMP/mmlupro.jsonl (300 items, stratified over categories, seed 0) and
TMP/supergpqa.jsonl (300 items, stratified over difficulty, seed 0). Each row:
{"question", "options": [...], "answer": "<letter>", "meta": {...}}
"""
import collections
import json
import random

from datasets import load_dataset

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
LET = "ABCDEFGHIJ"


def stratified(rows, key, n, seed=0):
    random.seed(seed)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out, per = [], max(1, n // len(groups))
    for g in sorted(groups):
        out += random.sample(groups[g], min(per, len(groups[g])))
    rest = [r for r in rows if r not in out]
    random.shuffle(rest)
    return out + rest[: max(0, n - len(out))]


mp = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
rows = [{"question": r["question"], "options": list(r["options"]), "answer": r["answer"], "meta": {"category": r["category"], "id": r["question_id"]}}
        for r in mp if r["answer"] in LET[: len(r["options"])]]
sub = stratified(rows, lambda r: r["meta"]["category"], 300)
with open(f"{TMP}/mmlupro.jsonl", "w") as f:
    for r in sub:
        f.write(json.dumps(r) + "\n")
print(f"mmlupro: {len(sub)} items over {len(set(r['meta']['category'] for r in sub))} categories; options/item median "
      f"{sorted(len(r['options']) for r in sub)[len(sub)//2]}")

sg = load_dataset("m-a-p/SuperGPQA", split="train")
rows = [{"question": r["question"], "options": list(r["options"]), "answer": r["answer_letter"],
         "meta": {"discipline": r["discipline"], "difficulty": r["difficulty"], "id": r["uuid"]}}
        for r in sg if r["answer_letter"] in LET[: len(r["options"])]]
sub = stratified(rows, lambda r: r["meta"]["difficulty"], 300)
with open(f"{TMP}/supergpqa.jsonl", "w") as f:
    for r in sub:
        f.write(json.dumps(r) + "\n")
print(f"supergpqa: {len(sub)} items; difficulty {collections.Counter(r['meta']['difficulty'] for r in sub)}; options median "
      f"{sorted(len(r['options']) for r in sub)[len(sub)//2]}")
