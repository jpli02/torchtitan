"""LiveBench language (ICLR 2025): connections (100), plot_unscrambling (40), typos (50) -> TMP/livebench_lang.jsonl
Row: {"question_id", "task", "question" (official prompt), "answer" (ground truth)}; shuffled (seed 0). --task lbl"""
import collections
import json
import random

from datasets import load_dataset

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
ds = load_dataset("livebench/language", split="test")
rows = []
for r in ds:
    t = r["turns"] if isinstance(r["turns"], list) else json.loads(r["turns"])
    rows.append({"question_id": r["question_id"], "task": r["task"], "question": t[0], "answer": r["ground_truth"]})
random.seed(0)
random.shuffle(rows)
with open(f"{TMP}/livebench_lang.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"livebench_lang: {len(rows)} {dict(collections.Counter(r['task'] for r in rows))}")
for t in ("typos", "plot_unscrambling"):
    ex = next(r for r in rows if r["task"] == t)
    print(f"   {t} prompt head: {ex['question'][:300]!r}")
