"""Prepare LiveCodeBench (code_generation_lite test5+test6) as plain JSONL with decoded tests.

Private tests are stored as base64(zlib(pickle(json_str))). They are decoded with a
RESTRICTED unpickler that refuses every global (the payload is a plain JSON string),
so no code from the dataset is executed.

Usage: python lcb_prep.py RAW_DIR OUT_DIR
Writes lcb_all.jsonl (342 problems) and lcb_em120.jsonl (60 easy + 60 medium, seed 0),
fields: question_id, difficulty, platform, contest_date, question_content, starter_code,
func_name, tests=[{input, output, testtype}].
"""
import base64
import collections
import io
import json
import pickle
import random
import sys
import zlib

raw, out = sys.argv[1], sys.argv[2]


class NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def decode_private(s):
    if not s:
        return []
    data = zlib.decompress(base64.b64decode(s))
    obj = NoGlobals(io.BytesIO(data)).load()
    return json.loads(obj) if isinstance(obj, str) else obj


rows = []
for fn in ("test5.jsonl", "test6.jsonl"):
    for line in open(f"{raw}/{fn}"):
        r = json.loads(line)
        pub = json.loads(r["public_test_cases"]) if isinstance(r["public_test_cases"], str) else r["public_test_cases"]
        priv = decode_private(r.get("private_test_cases", ""))
        meta = json.loads(r["metadata"]) if isinstance(r.get("metadata"), str) else (r.get("metadata") or {})
        rows.append({"question_id": r["question_id"], "difficulty": r["difficulty"], "platform": r["platform"],
                     "contest_date": r["contest_date"], "question_content": r["question_content"],
                     "starter_code": r.get("starter_code", ""), "func_name": meta.get("func_name"),
                     "tests": pub + priv})
print(f"problems: {len(rows)}; tests/problem median {sorted(len(r['tests']) for r in rows)[len(rows)//2]}; "
      f"types {collections.Counter(t['testtype'] for r in rows for t in r['tests'])}")
with open(f"{out}/lcb_all.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
random.seed(0)
sub = random.sample([r for r in rows if r["difficulty"] == "easy"], 60) + random.sample([r for r in rows if r["difficulty"] == "medium"], 60)
with open(f"{out}/lcb_em120.jsonl", "w") as f:
    for r in sub:
        f.write(json.dumps(r) + "\n")
print(f"wrote {out}/lcb_all.jsonl and {out}/lcb_em120.jsonl ({len(sub)} rows, "
      f"{sum(1 for r in sub if r['starter_code'].strip())} functional)")
