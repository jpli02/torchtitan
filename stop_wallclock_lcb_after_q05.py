"""Wait until the GPU 3 LiveCodeBench true-exit chain (queue_wallclock_lcb.sh -> wallclock_lcb.sh) has graded
gate0.5, then stop the WHOLE process tree (queue shell, chain shell, and any depth_eval it started for the
gate0.3 / gate0.7 points, which GPU 5 runs instead via wallclock_lcb2.sh). Lesson from the 11-02 stopper:
killing the shell and its direct child leaves the grandchild python running."""
import os
import signal
import time

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
REP = f"{TMP}/wallclock_lcb.txt"


def proc_list():
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            env = open(f"/proc/{pid}/environ", "rb").read().replace(b"\0", b"\n").decode(errors="replace")
        except Exception:
            continue
        out.append((int(pid), cmd, env))
    return out


end = time.time() + 12 * 3600
while time.time() < end:
    try:
        if any(line.startswith("gate0.5 GRADED") for line in open(REP)):
            break
    except FileNotFoundError:
        pass
    time.sleep(60)
else:
    raise SystemExit("gate0.5 never graded; not stopping")

victims = []
for pid, cmd, env in proc_list():
    if pid == os.getpid():
        continue
    if "queue_wallclock_lcb.sh" in cmd or "bash wallclock_lcb.sh" in cmd:
        victims.append((pid, cmd[:80]))
    elif "depth_eval.py" in cmd and ("wcl_gate0.3.jsonl" in cmd or "wcl_gate0.7.jsonl" in cmd) and "CUDA_VISIBLE_DEVICES=3" in env:
        victims.append((pid, cmd[:80]))
for pid, cmd in victims:  # shells first (so nothing new starts), then the generation
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"killed {pid}: {cmd}", flush=True)
    except ProcessLookupError:
        pass
time.sleep(5)
left = [p for p in proc_list() if "depth_eval.py" in p[1] and "wcl_gate0" in p[1] and "CUDA_VISIBLE_DEVICES=3" in p[2]]
for pid, cmd, _ in left:
    os.kill(pid, signal.SIGKILL)
    print(f"force-killed {pid}", flush=True)
with open(REP, "a") as f:
    f.write(f"GPU 3 chain stopped after gate0.5 {time.strftime('%m-%d %H:%M')} (gate0.3/gate0.7 run on GPU 5)\n")
print("done", flush=True)
