"""GSM-Symbolic (Apple, ICLR 2025; arXiv 2410.05229) as JSONL for depth_eval --task gsm8k.

Writes TMP/gsm_symbolic_{main,p1,p2}.jsonl: 300 items each, 3 instances per template (100 templates,
seed 0), rows {"question", "answer": "<final number>", "meta": {"id", "instance", "original_id"}}.
main = GSM8K templates with names/numbers resampled; p1/p2 = one/two extra clauses (harder).
"""
import collections
import json
import random
import sys

from datasets import load_dataset

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
PER_TEMPLATE = 3

for cfg in sys.argv[1:] or ["main", "p1", "p2"]:
    ds = load_dataset("apple/GSM-Symbolic", cfg, split="test")
    by_t = collections.defaultdict(list)
    for r in ds:
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        by_t[r["original_id"]].append({"question": r["question"], "answer": ans,
                                       "meta": {"id": r["id"], "instance": r["instance"], "original_id": r["original_id"]}})
    random.seed(0)
    rows = []
    for t in sorted(by_t):
        rows += random.sample(by_t[t], min(PER_TEMPLATE, len(by_t[t])))
    random.shuffle(rows)
    with open(f"{TMP}/gsm_symbolic_{cfg}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"gsm_symbolic_{cfg}: {len(rows)} items from {len(by_t)} templates ({len(ds)} available)")
