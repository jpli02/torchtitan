"""Print the schema and one truncated row of each newly downloaded candidate benchmark (read-only)."""
import collections
import glob
import json

import pyarrow as pa
import pyarrow.parquet as pq

R = "/home/jli199/boptim_scratch/more_bench_raw"


def show(name, rows, keys=None):
    print(f"\n=== {name}: {len(rows)} rows; keys {list(rows[0].keys())}")
    r = rows[0]
    for k, v in r.items():
        if keys and k not in keys:
            continue
        print(f"   {k}: {str(v)[:230]!r}")


rows = pq.read_table(f"{R}/lcb_exec/data/test-00000-of-00001.parquet").to_pylist()
show("LCB code execution (execution-v2)", rows)
print("   output lengths (chars) p50/p90/max:", sorted(len(str(r["output"])) for r in rows)[len(rows) // 2], sorted(len(str(r["output"])) for r in rows)[int(len(rows) * 0.9)], max(len(str(r["output"])) for r in rows))

rows = pq.read_table(f"{R}/lcb_testgen/test.parquet").to_pylist()
show("LCB test output prediction (test_generation)", rows)

for cfg in ("tot_arithmetic", "tot_semantic"):
    rows = json.load(open(f"{R}/tot/{cfg}/test.json"))
    if isinstance(rows, dict):
        rows = rows.get("data") or list(rows.values())[0]
    show(f"Test of Time {cfg}", rows)
    for key in ("question_type", "graph_gen_algorithm"):
        if key in rows[0]:
            print("   ", key, dict(collections.Counter(r[key] for r in rows)))

files = sorted(glob.glob(f"{R}/mmlu_redux/*/data-00000-of-00001.arrow"))
allrows = []
for f in files:
    with pa.ipc.open_stream(f) as rd:
        t = rd.read_all()
    for r in t.to_pylist():
        r["subject"] = f.split("/")[-2]
        allrows.append(r)
show(f"MMLU-Redux 2.0 ({len(files)} subjects)", allrows)
print("   error_type:", dict(collections.Counter(r.get("error_type") for r in allrows)))

rows = pq.read_table(f"{R}/gsm8k_platinum/main/test-00000-of-00001.parquet").to_pylist()
show("GSM8K-Platinum", rows)
if "cleaning_status" in rows[0]:
    print("   cleaning_status:", dict(collections.Counter(r["cleaning_status"] for r in rows)))

rows = [json.loads(l) for l in open(f"{R}/tablebench/TableBench.jsonl") if l.strip()]
show("TableBench", rows)
print("   qtype:", dict(collections.Counter(r.get("qtype") for r in rows)), "| qsubtype:", dict(collections.Counter(r.get("qsubtype") for r in rows).most_common(8)))
print("   table chars p50/p90/max:", sorted(len(json.dumps(r.get("table"))) for r in rows)[len(rows) // 2], sorted(len(json.dumps(r.get("table"))) for r in rows)[int(len(rows) * 0.9)], max(len(json.dumps(r.get("table"))) for r in rows))
