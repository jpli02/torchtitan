"""Non-coding 2025 benchmarks as JSONL for depth_eval.

  TMP/livebench_math.jsonl  LiveBench math (ICLR 2025): AMPS_Hard (150, LaTeX/numeric boxed answers) + math_comp (146, AMC/AIME-style
                            multiple choice whose answer is a letter); the olympiad masking task is left out. --task lbm
  TMP/livebench_da.jsonl    LiveBench data_analysis (150): tablereformat / cta / tablejoin, official prompts. --task lbd
  TMP/omnimath_easy.jsonl   Omni-MATH (ICLR 2025) problems with difficulty <= 5, 150 sampled (seed 0), {"question","answer"}. --task math
  TMP/aime_hmmt.jsonl       AIME 2024 (30) + AIME 2025 (30) + HMMT Feb 2025 (30): integer answers. --task gsm8k
"""
import collections
import json
import random

from datasets import load_dataset

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"


def turns0(r):
    t = r["turns"] if isinstance(r["turns"], list) else json.loads(r["turns"])
    return t[0]


# LiveBench math
ds = load_dataset("livebench/math", split="test")
rows = [{"question_id": r["question_id"], "task": r["task"], "subtask": r["subtask"], "question": turns0(r), "answer": r["ground_truth"]}
        for r in ds if r["task"] in ("AMPS_Hard", "math_comp")]
random.seed(0)
random.shuffle(rows)
with open(f"{TMP}/livebench_math.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"livebench_math: {len(rows)} {dict(collections.Counter(r['task'] for r in rows))}; e.g. answer {rows[0]['answer'][:40]!r} | prompt tail {rows[0]['question'][-160:]!r}")

# LiveBench data analysis
ds = load_dataset("livebench/data_analysis", split="test")
rows = [{"question_id": r["question_id"], "task": r["task"], "question": turns0(r), "answer": r["ground_truth"]} for r in ds]
random.seed(0)
random.shuffle(rows)
with open(f"{TMP}/livebench_da.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"livebench_da: {len(rows)} {dict(collections.Counter(r['task'] for r in rows))}")
for t in ("tablereformat", "cta", "tablejoin"):
    ex = next(r for r in rows if r["task"] == t)
    print(f"   {t}: prompt tail {ex['question'][-140:]!r} | answer {ex['answer'][:100]!r}")

# Omni-MATH easy tiers
ds = load_dataset("KbsdJames/Omni-MATH", split="test")
easy = [{"question": r["problem"], "answer": r["answer"], "meta": {"difficulty": r["difficulty"], "domain": str(r["domain"])[:60], "source": r["source"]}}
        for r in ds if r["difficulty"] is not None and float(r["difficulty"]) <= 5.0 and r["answer"] and len(str(r["answer"])) <= 60]
random.seed(0)
random.shuffle(easy)
sub = easy[:150]
with open(f"{TMP}/omnimath_easy.jsonl", "w") as f:
    for r in sub:
        f.write(json.dumps(r) + "\n")
print(f"omnimath_easy: {len(sub)} of {len(easy)} problems with difficulty <= 5 (of {len(ds)}); difficulties {dict(collections.Counter(r['meta']['difficulty'] for r in sub))}")

# AIME 2024 + AIME 2025 + HMMT Feb 2025
rows = []
for r in load_dataset("HuggingFaceH4/aime_2024", split="train"):
    rows.append({"question": r["problem"], "answer": str(r["answer"]), "meta": {"set": "aime2024", "id": str(r["id"])}})
for r in load_dataset("MathArena/aime_2025", split="train"):
    rows.append({"question": r["problem"], "answer": str(r["answer"]), "meta": {"set": "aime2025", "id": str(r["problem_idx"])}})
for r in load_dataset("MathArena/hmmt_feb_2025", split="train"):
    rows.append({"question": r["problem"], "answer": str(r["answer"]), "meta": {"set": "hmmt_feb2025", "id": str(r["problem_idx"])}})
with open(f"{TMP}/aime_hmmt.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"aime_hmmt: {len(rows)} problems {dict(collections.Counter(r['meta']['set'] for r in rows))}; non-integer answers: {sum(not r['answer'].strip().lstrip('-').isdigit() for r in rows)}")
