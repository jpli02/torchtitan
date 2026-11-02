"""Grade BigCodeBench generations: run each task's unittest suite against the generated code.
RUN INSIDE THE OFFICIAL IMAGE (stdlib only here; the image has every task library):

  docker run --rm --network none --memory 6g --cpus 4 -v $DIR:/w --entrypoint python \
      bigcodebench/bigcodebench-evaluate:latest /w/bcb_grade.py /w/bcb_hard.jsonl /w/gen.jsonl /w/gen.graded.json

Per task: solution code + "\n\n" + test code are written to one file and executed with
`python -m unittest` in a fresh temp working directory, 60 s limit (BigCodeBench's harness uses
a per-task limit of the same order). pass = exit code 0 with no failures/errors.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

TIMEOUT = float(os.environ.get("BCB_TIMEOUT", "60"))
data_path, gen_path, out_path = sys.argv[1:4]
tasks = {r["task_id"]: r for r in (json.loads(l) for l in open(data_path) if l.strip())}
gens = [json.loads(l) for l in open(gen_path) if l.strip()]
summary = gens[0].get("summary") if "summary" in gens[0] else None
gens = [g for g in gens if "task_id" in g]

results = []
for g in gens:
    t = tasks.get(g["task_id"])
    code = g.get("pred")
    rec = {"task_id": g["task_id"]}
    if t is None or not code:
        rec.update({"passed": False, "reason": "no code" if t else "unknown task"})
        results.append(rec)
        continue
    with tempfile.TemporaryDirectory() as wd:
        path = os.path.join(wd, "sol_test.py")
        with open(path, "w") as f:
            f.write(code.rstrip() + "\n\n" + t["test"] + "\n")
        try:
            p = subprocess.run([sys.executable, "-m", "unittest", "-q", "sol_test"], cwd=wd, capture_output=True, text=True,
                               timeout=TIMEOUT, env={**os.environ, "PYTHONPATH": wd, "MPLBACKEND": "Agg"})
            out = (p.stderr or "") + (p.stdout or "")
            m = re.search(r"Ran (\d+) tests?", out)
            n_tests = int(m.group(1)) if m else 0
            passed = p.returncode == 0 and n_tests > 0
            reason = "" if passed else ("no tests ran" if n_tests == 0 else out.strip().splitlines()[-1][:120])
        except subprocess.TimeoutExpired:
            passed, n_tests, reason = False, 0, "timeout"
    rec.update({"passed": passed, "n_tests": n_tests, "reason": reason})
    results.append(rec)
    print(f"  {g['task_id']}: {'PASS' if passed else 'fail'} {reason}", flush=True)

n = len(results)
acc = sum(r["passed"] for r in results) / max(n, 1)
out = {"summary": summary, "n": n, "pass@1": acc, "no_code": sum(1 for r in results if r.get("reason") == "no code"),
       "timeouts": sum(1 for r in results if r.get("reason") == "timeout"), "results": results}
json.dump(out, open(out_path, "w"), indent=1)
print(f"GRADED n={n} pass@1={acc:.3f} no_code={out['no_code']} timeouts={out['timeouts']}", flush=True)
