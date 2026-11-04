"""Grade HumanEval Pro / MBPP Pro generations by running each solution with its assert tests. RUN INSIDE A SANDBOX:
  docker run --rm --network none --memory 4g --cpus 4 -v $DIR:/w python:3.12-slim \
      python /w/codepro_grade.py /w/codepro.jsonl /w/GEN.jsonl /w/OUT.graded.json
pass = the program (solution code + test_code asserts) exits 0 within 10 s."""
import json
import os
import subprocess
import sys
import tempfile

TIMEOUT = 10.0
data_path, gen_path, out_path = sys.argv[1:4]
problems = {r["task_id"]: r for r in (json.loads(l) for l in open(data_path) if l.strip())}
gens = [json.loads(l) for l in open(gen_path) if l.strip()]
summary = gens[0].get("summary") if "summary" in gens[0] else None
gens = [g for g in gens if "task_id" in g]

results = []
for g in gens:
    prob = problems.get(g["task_id"])
    code = g.get("pred")
    rec = {"task_id": g["task_id"], "source": g.get("source")}
    if prob is None or not code:
        rec.update({"passed": False, "reason": "no code" if prob else "unknown problem"})
        results.append(rec)
        continue
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + prob["test_code"] + "\n")
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=TIMEOUT)
        ok = p.returncode == 0
        reason = "" if ok else ("assertion/runtime error: " + (p.stderr.strip().splitlines() or [""])[-1][:120])
    except subprocess.TimeoutExpired:
        ok, reason = False, "timeout"
    finally:
        os.unlink(path)
    rec.update({"passed": ok, "reason": reason})
    results.append(rec)

n = len(results)
acc = sum(r["passed"] for r in results) / max(n, 1)
by_src = {}
for r in results:
    d = by_src.setdefault(r["source"], [0, 0])
    d[0] += r["passed"]
    d[1] += 1
out = {"summary": summary, "n": n, "pass@1": acc, "by_source": {k: f"{v[0]}/{v[1]}" for k, v in by_src.items()},
       "no_code": sum(1 for r in results if r["reason"] == "no code"), "timeouts": sum(1 for r in results if r["reason"] == "timeout"),
       "results": results}
json.dump(out, open(out_path, "w"), indent=1)
print(f"GRADED n={n} pass@1={acc:.3f} by_source={out['by_source']} no_code={out['no_code']} timeouts={out['timeouts']}", flush=True)
