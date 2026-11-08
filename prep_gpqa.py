"""GPQA Diamond (198) -> TMP/gpqa_diamond.jsonl in depth_eval --task mc format, from the ungated hendrydong/gpqa_diamond_mc mirror
(the original Idavidrein/gpqa is gated). Row: {"question", "options": [4 texts], "answer": letter, "meta": {"domain"}}."""
import json, re, collections
import pyarrow.parquet as pq
SRC = "/home/jli199/boptim_scratch/dory_suite_raw/gpqa_hd/data/test-00000-of-00001.parquet"
OUT = "/home/jli199/.claude/jobs/c0d2da0a/tmp/gpqa_diamond.jsonl"
rows = pq.read_table(SRC).to_pylist()
out = []
for r in rows:
    p = r["problem"]
    p = re.sub(r"\n*Please write your final answer in the form of .*$", "", p, flags=re.S).rstrip()
    m = re.search(r"\n\(A\) (.*)\n\(B\) (.*)\n\(C\) (.*)\n\(D\) (.*)$", p, flags=re.S)
    assert m, p[-300:]
    opts = [m.group(k).strip() for k in range(1, 5)]
    assert all("\n(" not in o for o in opts[:3]), opts
    q = p[: m.start()].strip()
    ans = re.search(r"\\boxed\{([ABCD])\}", r["solution"]).group(1)
    out.append({"question": q, "options": opts, "answer": ans, "meta": {"domain": r["domain"]}})
with open(OUT, "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(len(out), collections.Counter(o["answer"] for o in out), collections.Counter(o["meta"]["domain"] for o in out))
print(json.dumps(out[0])[:400])
