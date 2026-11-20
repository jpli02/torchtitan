"""Real-execution LiveCodeBench score of a model directory under one router: the same generation (depth_eval.py --task lcb,
greedy, batch 1, forced ```python block, true decode-time early exit) and the same sandboxed grader as the router searches
(boptim_lcb/ouro_lcb_router.py), without the speed calibration; the cost is reported as executed loops per decoded token.

The 342 problems are cut into chunks that worker processes on several cards pull from a queue, so cards of different speed
share the work: --cards "0:2,8:1,9:1" starts two workers on card 0 and one on each of cards 8 and 9; ":cap" after a worker
count caps each of that card's workers to that many GB (a card whose memory is mostly another user's), e.g. "8:1:9.3". A chunk
that fails on a capped card (a long prompt) is retried on an uncapped one.

  python randrouter/lcb_real_eval.py --ckpt DIR --thr 0.3,0.3,0.3 --tag NAME [--cards 0:2] [--chunk 24] [--data FILE] [--n 342]
Writes <OUT>/<tag>/{chunk_*.jsonl/.log, gen.jsonl, graded.json, record.json}; a finished record is not recomputed.
"""
import argparse
import json
import os
import queue
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
ap.add_argument("--cards", default="0:2")
ap.add_argument("--chunk", type=int, default=24)
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--cfg", action="append", default=[], help="extra model setting key=value for depth_eval, e.g. decode_fill_maps=/path.pt (exit-lab directory)")
args = ap.parse_args()

thr = [float(v) for v in args.thr.split(",")]
thr = (thr * 3 if len(thr) == 1 else thr)[:3] + [1.0]
run_dir = Path(args.out) / args.tag
run_dir.mkdir(parents=True, exist_ok=True)
if (run_dir / "record.json").exists():
    print("already done:", json.dumps(json.load(open(run_dir / "record.json"))))
    sys.exit(0)

workers = []  # (card, cap_gb or None)
for part in args.cards.split(","):
    f = part.split(":")
    workers += [(f[0], float(f[2]) if len(f) > 2 else None)] * int(f[1] if len(f) > 1 else 1)
chunks = [(lo, min(args.n, lo + args.chunk)) for lo in range(0, args.n, args.chunk)]
todo: "queue.Queue" = queue.Queue()
big_only: "queue.Queue" = queue.Queue()
results, lock = {}, threading.Lock()
for c in chunks:
    todo.put(c)


def run_chunk(card, cap, lo, hi):
    out, log = run_dir / f"chunk_{lo:03d}.jsonl", run_dir / f"chunk_{lo:03d}.log"
    if out.exists() and log.exists():  # a chunk finished by an earlier, interrupted run
        for line in open(log, errors="replace"):
            if line.startswith("RESULT "):
                return json.loads(line[7:])
    cmd = [PY, str(WT / "depth_eval.py"), args.ckpt, args.data, "--task", "lcb", "--steps", "4", "--n", str(hi - lo),
           "--seed_offset", str(args.offset + lo), "--batch", "1", "--max_new", str(args.max_new), "--assistant_prefix", "```python\\n",
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
    while True:
        try:
            c = big_only.get_nowait() if not cap else todo.get_nowait()
        except queue.Empty:
            try:
                c = todo.get_nowait()
            except queue.Empty:
                if cap or (big_only.empty() and done_event.is_set()):
                    return
                time.sleep(20)
                continue
        t0 = time.time()
        res = None
        try:
            res = run_chunk(card, cap, *c)
        except Exception as e:  # noqa: BLE001
            print(f"chunk {c} on card {card} raised {type(e).__name__}: {str(e)[:100]}", flush=True)
        with lock:
            if res is None:
                fails[c] = fails.get(c, 0) + 1
                if fails[c] <= 2:
                    (big_only if cap else todo).put(c)
                    print(f"chunk {c} failed on card {card}{' (capped)' if cap else ''}; queued again", flush=True)
                else:
                    results[c] = None
            else:
                results[c] = res
                print(f"chunk {c} done on card {card} in {time.time() - t0:.0f}s ({len(results)}/{len(chunks)})", flush=True)
            if len(results) == len(chunks):
                done_event.set()


fails = {}
done_event = threading.Event()
t_start = time.time()
threads = [threading.Thread(target=worker, args=w) for w in workers]
for t in threads:
    t.start()
for t in threads:
    t.join()
# capped workers leave when the common queue is empty; anything still queued for big cards is run here
while not big_only.empty() or not todo.empty():
    uncapped = [w for w in workers if not w[1]] or workers[:1]
    threads = [threading.Thread(target=worker, args=(w[0], None)) for w in uncapped]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
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
