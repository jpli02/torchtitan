"""Training pool for the random-router experiment: LiveCodeBench code generation, releases v1 to v4 (713 problems, contest
dates 2023-05 to 2024-10). The evaluation set of the router searches is releases v5 + v6 (342 problems, lcb_all.jsonl); the two
share no question id and no problem text (checked here), so nothing the model is trained on is a test problem.

Private tests are decoded with the same restricted unpickler as lcb_prep.py (no code from the dataset is executed), so that the
pool can also serve as a development set with real execution.

  python randrouter/prep_lcb_v1v4.py
Writes  <OUT>/lcb_v1v4.jsonl        713 rows, the lcb_all.jsonl fields plus "release" and "split" (train / val)
        <OUT>/lcb_v1v4_notests.jsonl  the same without the tests (small; what the generation and training scripts read)
Split: 100 validation problems, stratified by difficulty (seed 0); the other 613 train.
"""
import base64
import collections
import io
import json
import pickle
import random
import zlib

RAW = "/home/jli199/boptim_scratch/lcb_raw_v1v4"
TEST = "/home/jli199/.claude/jobs/c0d2da0a/tmp/lcb_all.jsonl"
OUT = "/home/jli199/boptim_scratch/randrouter"
N_VAL = 100


class NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def decode_private(s):
    if not s:
        return []
    obj = NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(s)))).load()
    return json.loads(obj) if isinstance(obj, str) else obj


test_rows = [json.loads(l) for l in open(TEST)]
test_ids = {r["question_id"] for r in test_rows}
test_text = {r["question_content"].strip() for r in test_rows}

rows, seen = [], set()
for rel, fn in enumerate(("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"), start=1):
    for line in open(f"{RAW}/{fn}"):
        r = json.loads(line)
        assert r["question_id"] not in test_ids and r["question_content"].strip() not in test_text, f"test problem in the pool: {r['question_id']}"
        if r["question_id"] in seen:
            continue
        seen.add(r["question_id"])
        pub = json.loads(r["public_test_cases"]) if isinstance(r["public_test_cases"], str) else r["public_test_cases"]
        try:
            priv = decode_private(r.get("private_test_cases", ""))
        except Exception as e:  # keep the problem; it only cannot be executed against the private tests
            print(f"private tests of {r['question_id']} not decoded: {type(e).__name__}")
            priv = []
        meta = json.loads(r["metadata"]) if isinstance(r.get("metadata"), str) else (r.get("metadata") or {})
        rows.append({"question_id": r["question_id"], "difficulty": r["difficulty"], "platform": r["platform"],
                     "contest_date": r["contest_date"], "question_content": r["question_content"],
                     "starter_code": r.get("starter_code", ""), "func_name": meta.get("func_name"),
                     "tests": pub + priv, "release": f"v{rel}"})

rng = random.Random(0)
by_diff = collections.defaultdict(list)
for i, r in enumerate(rows):
    by_diff[r["difficulty"]].append(i)
val = set()
for d, idx in sorted(by_diff.items()):
    k = round(N_VAL * len(idx) / len(rows))
    val.update(rng.sample(idx, k))
for i, r in enumerate(rows):
    r["split"] = "val" if i in val else "train"

with open(f"{OUT}/lcb_v1v4.jsonl", "w") as f, open(f"{OUT}/lcb_v1v4_notests.jsonl", "w") as g:
    for r in rows:
        f.write(json.dumps(r) + "\n")
        g.write(json.dumps({k: v for k, v in r.items() if k != "tests"}) + "\n")
c = collections.Counter((r["split"], r["difficulty"]) for r in rows)
print(f"problems {len(rows)}; split x difficulty {dict(sorted(c.items()))}; functional {sum(1 for r in rows if r['starter_code'].strip())}; "
      f"tests/problem median {sorted(len(r['tests']) for r in rows)[len(rows) // 2]}")
