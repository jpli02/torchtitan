"""LiveBench (ICLR 2025) reasoning and data_analysis categories as JSONL for depth_eval --task lbr (exact-match string answers).

  TMP/livebench_reasoning.jsonl : all questions of livebench/reasoning (zebra_puzzle, web_of_lies, spatial) with the
                                  official prompt text (turns[0]) and ground_truth.
Row: {"question_id", "task", "question", "answer", "release"}"""
import collections
import json

from datasets import load_dataset

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
ds = load_dataset("livebench/reasoning", split="test")
rows = []
for r in ds:
    turns = r["turns"] if isinstance(r["turns"], list) else json.loads(r["turns"])
    rows.append({"question_id": r["question_id"], "task": r["task"], "question": turns[0], "answer": r["ground_truth"],
                 "release": str(r.get("livebench_release_date", ""))})
with open(f"{TMP}/livebench_reasoning.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"livebench_reasoning: {len(rows)} questions {dict(collections.Counter(r['task'] for r in rows))}; "
      f"releases {dict(collections.Counter(r['release'][:10] for r in rows))}")
print("example answer:", rows[0]["answer"][:80], "| prompt tail:", rows[0]["question"][-200:].replace("\n", " "))
