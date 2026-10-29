"""Grade LiveCodeBench generations by executing them against the tests. RUN INSIDE A
SANDBOX (docker python:3.12-slim, --network none, memory/cpu limited):

  docker run --rm --network none --memory 4g --cpus 4 -v $DIR:/w python:3.12-slim \
      python /w/lcb_grade.py /w/lcb_em120.jsonl /w/depth_lcb_k4.jsonl /w/depth_lcb_k4.graded.json

stdin problems: run the program with each test input, compare stdout (trailing
whitespace per line stripped). Functional problems: exec the code, call
Solution().<func_name>(*args) with args parsed from the JSON-lines input, compare the
result to the JSON-parsed expected output. 6 s per test. pass = every test passes.
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile

TIMEOUT = 6.0
data_path, gen_path, out_path = sys.argv[1:4]
problems = {r["question_id"]: r for r in (json.loads(l) for l in open(data_path) if l.strip())}
gens = [json.loads(l) for l in open(gen_path) if l.strip()]
summary = gens[0].get("summary") if "summary" in gens[0] else None
gens = [g for g in gens if "question_id" in g]


def norm_out(s):
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def run_stdin(code, inp):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], input=inp, capture_output=True, text=True, timeout=TIMEOUT)
        return p.stdout, (p.returncode != 0)
    except subprocess.TimeoutExpired:
        return "", True
    finally:
        os.unlink(path)


def _call(code, func, args, q):
    try:
        import builtins  # noqa
        ns = {}
        exec(code, ns)  # noqa: S102 - sandboxed by the container
        cls = ns.get("Solution")
        fn = getattr(cls(), func) if cls is not None else ns[func]
        q.put(("ok", json.dumps(fn(*args), default=str)))
    except BaseException as e:  # noqa
        q.put(("err", repr(e)[:200]))


def run_functional(code, func, inp, timeout=TIMEOUT):
    args = [json.loads(line) for line in inp.strip().splitlines() if line.strip()]
    q = mp.Queue()
    p = mp.Process(target=_call, args=(code, func, args, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        return None, True
    try:
        status, val = q.get_nowait()
    except Exception:
        return None, True
    return (json.loads(val) if status == "ok" else None), status != "ok"


results = []
for g in gens:
    prob = problems.get(g["question_id"])
    code = g.get("pred")
    rec = {"question_id": g["question_id"], "difficulty": g.get("difficulty"), "n_tests": len(prob["tests"]) if prob else 0}
    if prob is None or not code:
        rec.update({"passed": False, "tests_passed": 0, "reason": "no code" if prob else "unknown problem"})
        results.append(rec)
        continue
    passed = 0
    reason = ""
    for t in prob["tests"]:
        if t["testtype"] == "stdin":
            out, err = run_stdin(code, t["input"])
            ok = (not err) and norm_out(out) == norm_out(t["output"])
        else:
            val, err = run_functional(code, prob.get("func_name"), t["input"])
            try:
                exp = json.loads(t["output"])
            except Exception:
                exp = t["output"]
            ok = (not err) and (val == exp or json.dumps(val, sort_keys=True) == json.dumps(exp, sort_keys=True))
        if ok:
            passed += 1
        else:
            reason = "runtime error/timeout" if err else "wrong answer"
            break
    rec.update({"passed": passed == len(prob["tests"]), "tests_passed": passed, "reason": reason})
    results.append(rec)
    print(f"  {g['question_id']}: {'PASS' if rec['passed'] else 'fail'} {passed}/{len(prob['tests'])} {reason}", flush=True)

n = len(results)
acc = sum(r["passed"] for r in results) / max(n, 1)
by_diff = {}
for r in results:
    d = by_diff.setdefault(r["difficulty"], [0, 0])
    d[0] += r["passed"]
    d[1] += 1
out = {"summary": summary, "n": n, "pass@1": acc, "by_difficulty": {k: f"{v[0]}/{v[1]}" for k, v in by_diff.items()},
       "no_code": sum(1 for r in results if r["reason"] == "no code"), "results": results}
json.dump(out, open(out_path, "w"), indent=1)
print(f"GRADED n={n} pass@1={acc:.3f} by_difficulty={out['by_difficulty']} no_code={out['no_code']}", flush=True)
