"""More 2025 benchmarks as JSONL for depth_eval.

  TMP/zebralogic.jsonl   WildEval/ZebraLogic grid_mode: 300 puzzles stratified over the 25 sizes (2x2 .. 6x6), seed 0.
                         Row: {"id", "size", "puzzle", "solution": {"House 1": {"Name": ..}, ...}, "header": [...]}
  TMP/codepro.jsonl      CodeEval-Pro HumanEval Pro (164) + MBPP Pro (378): self-invoking code generation (ACL 2025 findings).
                         Row: {"task_id", "source", "problem" (new_problem: signature + docstring of the new function, which must
                         call a solution of the raw problem), "raw_problem", "raw_solution", "test_code" (asserts)}
  TMP/codemmlu.jsonl     Fsoft-AIC/CodeMMLU (ICLR 2025): 300 items stratified over the 8 configs, for depth_eval --task mc.
"""
import collections
import json
import random

from datasets import get_dataset_config_names, load_dataset

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"


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


# ---------------------------------------------------------------- ZebraLogic
zb = load_dataset("WildEval/ZebraLogic", "grid_mode", split="test")
rows = []
for r in zb:
    sol = r["solution"] if isinstance(r["solution"], dict) else json.loads(r["solution"].replace("'", '"'))
    header, grid = sol["header"], sol["rows"]
    houses = {f"House {row[0]}": {header[j]: row[j] for j in range(1, len(header))} for row in grid}
    rows.append({"id": r["id"], "size": r["size"], "puzzle": r["puzzle"], "header": header[1:], "solution": houses})
sub = stratified(rows, lambda r: r["size"], 300)
with open(f"{TMP}/zebralogic.jsonl", "w") as f:
    for r in sub:
        f.write(json.dumps(r) + "\n")
sizes = collections.Counter(r["size"] for r in sub)
print(f"zebralogic: {len(sub)} of {len(rows)} puzzles over {len(sizes)} sizes; e.g. {dict(list(sizes.items())[:5])}")

# ---------------------------------------------------------------- HumanEval Pro + MBPP Pro
out = []
for name, src in [("CodeEval-Pro/humaneval-pro", "humaneval_pro"), ("CodeEval-Pro/mbpp-pro", "mbpp_pro")]:
    ds = load_dataset(name, split="train")
    for r in ds:
        out.append({"task_id": f"{src}/{r['id']}", "source": src, "problem": r["new_problem"], "raw_problem": r["raw_problem"],
                    "raw_solution": r["raw_solution"], "test_code": r["test_code"]})
with open(f"{TMP}/codepro.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"codepro: {len(out)} problems ({collections.Counter(r['source'] for r in out)})")

# ---------------------------------------------------------------- CodeMMLU
rows = []
for cfg in get_dataset_config_names("Fsoft-AIC/CodeMMLU"):
    ds = load_dataset("Fsoft-AIC/CodeMMLU", cfg, split="test")
    for r in ds:
        ch = r["choices"] if isinstance(r["choices"], list) else json.loads(r["choices"])
        if r["answer"] not in "ABCDEFGHIJ"[: len(ch)]:
            continue
        rows.append({"question": r["question"], "options": list(ch), "answer": r["answer"], "meta": {"config": cfg, "id": r["task_id"]}})
sub = stratified(rows, lambda r: r["meta"]["config"], 300)
with open(f"{TMP}/codemmlu.jsonl", "w") as f:
    for r in sub:
        f.write(json.dumps(r) + "\n")
print(f"codemmlu: {len(sub)} of {len(rows)} items over {len(set(r['meta']['config'] for r in sub))} configs")
