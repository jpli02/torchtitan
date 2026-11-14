"""More 2025-venue benchmarks as JSONL for depth_eval (run with the torchtitan venv: needs pyarrow).

  TMP/lcb_exec.jsonl      LiveCodeBench code execution (ICLR 2025; livecodebench/execution-v2), all 479: {"code", "call", "output", "task"}
                          -> depth_eval --task crux (the call is given in full, e.g. distinctDifferenceArray(a = [1, 2, 3]))
  TMP/lcb_testout.jsonl   LiveCodeBench test-output prediction (ICLR 2025; livecodebench/test_generation), all 442:
                          {"question", "starter_code", "call", "output", "task"} -> depth_eval --task lcbt
  TMP/mmlu_redux.jsonl    MMLU-Redux 2.0 (NAACL 2025; edinburgh-dawg/mmlu-redux-2.0): 35 questions per subject from the rows the
                          re-annotation marked "ok" (1,995 of 5,330), seed 0 -> depth_eval --task mc
  TMP/tot_arith.jsonl     Test of Time (ICLR 2025; baharef/ToT) arithmetic: 50 per question type (350) -> depth_eval --task tot
  TMP/tot_sem.jsonl       Test of Time semantic: 40 per question type among prompts under 12,000 characters (320) -> --task tot
  TMP/tablebench.jsonl    TableBench (AAAI 2025): NumericalReasoning + FactChecking (493; exact-match answers) -> depth_eval --task tbench
"""
import collections
import glob
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq

R = "/home/jli199/boptim_scratch/more_bench_raw"
TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"


def dump(name, rows):
    with open(f"{TMP}/{name}", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)} rows")


# ---- LiveCodeBench code execution
rows = pq.read_table(f"{R}/lcb_exec/data/test-00000-of-00001.parquet").to_pylist()
dump("lcb_exec.jsonl", [{"code": r["code"], "call": r["input"], "output": r["output"], "task": r["difficulty"], "question_id": f'{r["question_id"]}/{r["id"]}'} for r in rows])

# ---- LiveCodeBench test-output prediction
rows = pq.read_table(f"{R}/lcb_testgen/test.parquet").to_pylist()
out = []
for r in rows:
    t = json.loads(r["test"])[0]
    args_ = ", ".join(a for a in t["input"].split("\n") if a.strip())
    out.append({"question": r["question_content"], "starter_code": r["starter_code"], "call": f'{r["function_name"]}({args_})', "output": t["output"],
                "task": r["difficulty"], "question_id": f'{r["question_id"]}/{r["test_id"]}'})
dump("lcb_testout.jsonl", out)

# ---- MMLU-Redux 2.0
rng = random.Random(0)
out = []
for f in sorted(glob.glob(f"{R}/mmlu_redux/*/data-00000-of-00001.arrow")):
    with pa.ipc.open_stream(f) as rd:
        t = rd.read_all().to_pylist()
    ok = [r for r in t if r["error_type"] == "ok"]
    rng.shuffle(ok)
    subj = f.split("/")[-2]
    for r in ok[:35]:
        out.append({"question": r["question"], "options": list(r["choices"]), "answer": "ABCD"[int(r["answer"])], "meta": {"subject": subj}})
rng.shuffle(out)
dump("mmlu_redux.jsonl", out)

# ---- Test of Time
rng = random.Random(0)
a = json.load(open(f"{R}/tot/tot_arithmetic/test.json"))
by = collections.defaultdict(list)
for r in a:
    by[r["question_type"]].append(r)
out = []
for k in sorted(by):
    rng.shuffle(by[k])
    out += [{"prompt": r["question"], "label": r["label"], "task": k} for r in by[k][:50]]
rng.shuffle(out)
dump("tot_arith.jsonl", out)
s = json.load(open(f"{R}/tot/tot_semantic/test.json"))
by = collections.defaultdict(list)
for r in s:
    if len(r["prompt"]) < 12000:
        by[r["question_type"]].append(r)
out = []
for k in sorted(by):
    rng.shuffle(by[k])
    out += [{"prompt": r["prompt"], "label": r["label"], "task": k} for r in by[k][:40]]
rng.shuffle(out)
dump("tot_sem.jsonl", out)

# ---- TableBench (exact-match categories)
rows = [json.loads(l) for l in open(f"{R}/tablebench/TableBench.jsonl") if l.strip()]
out = []
for r in rows:
    if r["qtype"] not in ("NumericalReasoning", "FactChecking"):
        continue
    table = r["table"] if isinstance(r["table"], str) else json.dumps(r["table"])
    out.append({"table": table, "question": r["question"], "answer": str(r["answer"]), "task": r["qtype"], "subtask": r["qsubtype"]})
rng.shuffle(out)
dump("tablebench.jsonl", out)
