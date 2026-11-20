"""Real-execution LiveCodeBench score of a model directory under one router, version 2 of lcb_real_eval.py: the same
generation and grader, but the problems are sorted by prompt length so that small cards can take part.

Cards 8 and 9 have about 9 GB free next to another user's idle inference server: enough for the model (5.5 GB) and the 4-loop
cache of a problem with a short prompt, not of a long one. So the problems are sorted by prompt length and cut into chunks;
workers on capped cards ("8:1:8.6" = one worker on card 8, capped at 8.6 GB) take chunks from the SHORT end, workers on
uncapped cards ("0:3") from the LONG end, until they meet. A chunk that fails on a capped card goes to the uncapped workers;
if there is none, the leftovers run on --fallback when that card has room.

  python randrouter/lcb_real_eval2.py --ckpt DIR --thr 0.245,0.3,0.3 --tag NAME --cards 0:3,8:1:8.6,9:1:8.6 [--chunk 12]
        [--cfg decode_fill_maps=FILE] [--data FILE] [--n 342]
Writes <OUT>/<tag>/{data_sorted.jsonl, chunk_*.jsonl/.log, gen.jsonl, graded.json, record.json}; finished chunks and a finished
record are not recomputed. Cost is reported as executed loops per decoded token (no speed calibration).
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

WT = Path("/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
PY = "/home/jli199/boptim-agent/.venv/bin/python"
STAGE = Path("/srv/scratch/jli199/lcb_grade_stage_bo")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--thr", required=True, help="q1,q2,q3 (1,1,1 = full depth) or one value for all three")
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="/home/jli199/boptim_scratch/randrouter/real")
ap.add_argument("--data", default="/home/jli199/.claude/jobs/c0d2da0a/tmp/lcb_all.jsonl")
ap.add_argument("--n", type=int, default=342)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--cards", default="0:3")
ap.add_argument("--chunk", type=int, default=12)
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--cfg", action="append", default=[], help="extra model setting key=value for depth_eval, e.g. decode_fill_maps=/path.pt")
ap.add_argument("--fallback", default="0", help="card for chunks no worker could finish")
ap.add_argument("--fallback_free_mb", type=int, default=12000)
args = ap.parse_args()

thr = [float(v) for v in args.thr.split(",")]
thr = (thr * 3 if len(thr) == 1 else thr)[:3] + [1.0]
run_dir = Path(args.out) / args.tag
run_dir.mkdir(parents=True, exist_ok=True)
if (run_dir / "record.json").exists():
    print("already done:", json.dumps(json.load(open(run_dir / "record.json"))))
    sys.exit(0)

rows = [l for l in open(args.data) if l.strip()][args.offset:args.offset + args.n]
rows.sort(key=lambda l: (lambda r: len(r["question_content"]) + len(r.get("starter_code") or ""))(json.loads(l)))
sorted_path = run_dir / "data_sorted.jsonl"
if not sorted_path.exists():
    sorted_path.write_text("".join(l if l.endswith("\n") else l + "\n" for l in rows))
n = len(rows)
chunks = [(lo, min(n, lo + args.chunk)) for lo in range(0, n, args.chunk)]
workers = []  # (card, cap_gb or None)
for part in args.cards.split(","):
    f = part.split(":")
    workers += [(f[0], float(f[2]) if len(f) > 2 else None)] * int(f[1] if len(f) > 1 else 1)

lock = threading.Lock()
todo = collections.deque(chunks)     # short prompts on the left, long ones on the right
big, results, fails = [], {}, {}
inflight_capped = 0


def run_chunk(card, cap, lo, hi):
    out, log = run_dir / f"chunk_{lo:03d}.jsonl", run_dir / f"chunk_{lo:03d}.log"
    if out.exists() and log.exists():  # finished by an earlier, interrupted run
        for line in open(log, errors="replace"):
            if line.startswith("RESULT "):
                return json.loads(line[7:])
    cmd = [PY, str(WT / "depth_eval.py"), args.ckpt, str(sorted_path), "--task", "lcb", "--steps", "4", "--n", str(hi - lo),
           "--seed_offset", str(lo), "--batch", "1", "--max_new", str(args.max_new), "--assistant_prefix", "```python\\n",
           "--cfg", "decode_early_exit_threshold=" + json.dumps(thr), "--out", str(out)]
    for extra in args.cfg:
        cmd += ["--cfg", extra]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=card, PYTORCH_ALLOC_CONF="expandable_segments:True")
    if cap:
        env["DEPTH_EVAL_MEM_GB"] = str(cap)
    with open(log, "w") as lf:
        subprocess.run(cmd, cwd=str(WT), env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=6 * 3600)
    for line in open(log, errors="replace"):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    out.unlink(missing_ok=True)
    return None


def worker(card, cap):
    global inflight_capped
    while True:
        with lock:
            c = None
            if not cap and big:
                c = big.pop()
            elif todo:
                c = todo.popleft() if cap else todo.pop()
            if c is None:
                if cap or inflight_capped == 0:
                    return                      # nothing left for this worker (a capped chunk still running may come back to `big`)
            elif cap:
                inflight_capped += 1
        if c is None:
            time.sleep(15)
            continue
        t0 = time.time()
        res = None
        try:
            res = run_chunk(card, cap, *c)
        except Exception as e:  # noqa: BLE001
            print(f"chunk {c} on card {card} raised {type(e).__name__}: {str(e)[:100]}", flush=True)
        with lock:
            if cap:
                inflight_capped -= 1
            if res is None:
                fails[c] = fails.get(c, 0) + 1
                if cap or fails[c] <= 2:
                    (big if cap else todo).append(c)
                    print(f"chunk {c} failed on card {card}{' (capped)' if cap else ''}; queued for an uncapped worker", flush=True)
                else:
                    results[c] = None
            else:
                results[c] = res
                print(f"chunk {c} done on card {card} in {time.time() - t0:.0f}s ({len(results)}/{len(chunks)})", flush=True)


def free_mb(card):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", card],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out.split()[0])
    except Exception:
        return 0


t_start = time.time()
threads = [threading.Thread(target=worker, args=w) for w in workers]
for t in threads:
    t.start()
for t in threads:
    t.join()
left = list(big) + list(todo)
big.clear()
todo.clear()
for c in left:       # no uncapped worker took them: one at a time on the fallback card, when it has room
    while free_mb(args.fallback) < args.fallback_free_mb:
        time.sleep(60)
    res = None
    for _ in range(2):
        try:
            res = run_chunk(args.fallback, None, *c)
        except Exception as e:  # noqa: BLE001
            print(f"chunk {c} on the fallback card raised {type(e).__name__}: {str(e)[:100]}", flush=True)
        if res is not None:
            break
    results[c] = res
    print(f"chunk {c} {'done' if res else 'FAILED'} on fallback card {args.fallback} ({len(results)}/{len(chunks)})", flush=True)
if any(results.get(c) is None for c in chunks):
    print("FAILED chunks:", [c for c in chunks if results.get(c) is None])
    sys.exit(1)

with open(run_dir / "gen.jsonl", "w") as out:
    for lo, hi in chunks:
        for line in open(run_dir / f"chunk_{lo:03d}.jsonl"):
            if line.strip() and "question_id" in line:
                out.write(line if line.endswith("\n") else line + "\n")
rs = [results[c] for c in chunks]
n_tot = sum(r["n"] for r in rs)
ex_n = sum(r.get("exec_n", 0) for r in rs)
exec_loops = (sum(r["exec_loops"] * r["exec_n"] for r in rs if r.get("exec_n")) / ex_n) if ex_n else 4.0
label = f"rr_{args.tag}"
STAGE.mkdir(parents=True, exist_ok=True)
subprocess.run(["cp", str(WT / "lcb_grade.py"), str(STAGE / "lcb_grade.py")], check=False)
if not (STAGE / Path(args.data).name).exists():
    subprocess.run(["cp", args.data, str(STAGE / Path(args.data).name)], check=False)
subprocess.run(["cp", str(run_dir / "gen.jsonl"), str(STAGE / f"{label}.jsonl")], check=False)
grade_log = subprocess.run(
    ["docker", "run", "--rm", "--network", "none", "--memory", "12g", "--cpus", "4", "-v", f"{STAGE}:/w", "python:3.12-slim",
     "python", "/w/lcb_grade.py", f"/w/{Path(args.data).name}", f"/w/{label}.jsonl", f"/w/{label}.graded.json"],
    capture_output=True, text=True, timeout=3600).stdout
m = re.search(r"pass@1=([0-9.]+)", grade_log)
if not m:
    print("grading produced no pass@1:", grade_log[-400:])
    sys.exit(1)
subprocess.run(["cp", str(STAGE / f"{label}.graded.json"), str(run_dir / "graded.json")], check=False)
rec = {"ckpt": args.ckpt, "thresholds": thr, "cfg": args.cfg, "n": n_tot, "pass1_pct": 100.0 * float(m.group(1)), "exec_loops": exec_loops,
       "mean_tokens": sum(r["mean_tokens"] * r["n"] for r in rs) / n_tot, "trunc_rate": sum(r["trunc_rate"] * r["n"] for r in rs) / n_tot,
       "wall_s": time.time() - t_start, "cards": args.cards, "run_dir": str(run_dir)}
json.dump(rec, open(run_dir / "record.json", "w"), indent=1)
print("RECORD " + json.dumps(rec), flush=True)
