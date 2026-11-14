"""NUPA (ICLR 2025, number understanding and processing; HaotongYang/NUPA_text) -> TMP/nupa.jsonl for depth_eval --task nupa.
15 integer / float tasks x 3 digit lengths (6, 10, 14) x 20 items = 900 rows, seed 0. Each source item is one string
"<instruction> <question> = <answer>"; the prompt is everything up to and including the last " =", the gold answer what follows.
Row: {"prompt", "answer", "task": "<short task>/<digits>"}. The test split holds 1,000 items per task and length, so a disjoint
held-out file of any size can be drawn the same way (OFFSET).
  python prep_nupa.py [PER_CELL=20] [OFFSET=0] [OUT=TMP/nupa.jsonl]"""
import json
import random
import sys

SRC = "/home/jli199/boptim_scratch/more_bench_raw/nupa/test.json"
per_cell = int(sys.argv[1]) if len(sys.argv) > 1 else 20
offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
out = sys.argv[3] if len(sys.argv) > 3 else "/home/jli199/.claude/jobs/c0d2da0a/tmp/nupa.jsonl"
TASKS = [("add_Integer_Integer_Integer", "add_int"), ("sub_Integer_Integer_Integer", "sub_int"), ("multiply_easy_Integer_Integer_Integer", "mul_easy_int"),
         ("multiply_hard_Integer_Integer_Integer", "mul_hard_int"), ("floordiv_Integer_Integer_Integer", "floordiv_int"), ("mod_easy_Integer_Integer_Integer", "mod_easy_int"),
         ("max_hard_Integer_Integer_Integer", "max_hard_int"), ("digit_max_Integer_Integer_Integer", "digit_max_int"), ("digit_add_Integer_Integer_Integer", "digit_add_int"),
         ("get_digit_Integer_int_int", "get_digit_int"), ("length_Integer_none_int", "length_int"), ("count_Integer_int_int", "count_int"),
         ("add_Float_Float_Float", "add_float"), ("to_scient_Integer_none_ScientificNotation", "to_scient_int"), ("sig_Integer_int_ScientificNotation", "sig_int")]
DIGITS = ["6", "10", "14"]
data = json.load(open(SRC))
rng = random.Random(0)
rows = []
for name, short in TASKS:
    for d in DIGITS:
        items = list(data[name][d])
        rng.shuffle(items)
        for s in items[offset:offset + per_cell]:
            k = s.rfind(" = ")
            rows.append({"prompt": s[:k] + " =", "answer": s[k + 3:].strip(), "task": f"{short}/{d}"})
rng.shuffle(rows)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"{out}: {len(rows)} rows; example: {rows[0]}")
