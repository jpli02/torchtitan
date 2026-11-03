"""Generic chain stopper: wait until a report file has a line starting with MARK, then kill the whole process
tree of a running chain: every process whose cmdline contains SHELL_PAT and whose environment contains all of
ENV_MATCH (comma-separated key=value), plus any depth_eval.py whose cmdline contains KILL_PAT and whose
environment has CUDA_VISIBLE_DEVICES=GPU. Appends a note to the report.
  REPORT=... MARK='gate0.5_o120 GRADED' SHELL_PAT='bash wallclock_lcb2.sh' ENV_MATCH='OFFSET=120,GPU=2' \
  KILL_PAT='wcl_gate0.7_o120.jsonl' GPU=2 NOTE='GPU 2 chain stopped after gate0.5_o120' python stop_chain_after.py"""
import os
import signal
import time

REPORT = os.environ["REPORT"]
MARK = os.environ["MARK"]
SHELL_PAT = os.environ["SHELL_PAT"]
ENV_MATCH = [s for s in os.environ.get("ENV_MATCH", "").split(",") if s]
KILL_PAT = os.environ.get("KILL_PAT", "")
GPU = os.environ["GPU"]
NOTE = os.environ.get("NOTE", "chain stopped")
WAIT_H = float(os.environ.get("WAIT_H", "14"))


def procs():
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            env = open(f"/proc/{pid}/environ", "rb").read().replace(b"\0", b"\n").decode(errors="replace").split("\n")
        except Exception:
            continue
        yield int(pid), cmd, set(env)


end = time.time() + WAIT_H * 3600
while time.time() < end:
    try:
        if any(line.startswith(MARK) for line in open(REPORT)):
            break
    except FileNotFoundError:
        pass
    time.sleep(60)
else:
    raise SystemExit(f"{MARK} never appeared; not stopping")

victims = []
for pid, cmd, env in procs():
    if SHELL_PAT in cmd and all(m in env for m in ENV_MATCH):
        victims.append((pid, cmd[:80]))
    elif KILL_PAT and "depth_eval.py" in cmd and KILL_PAT in cmd and f"CUDA_VISIBLE_DEVICES={GPU}" in env:
        victims.append((pid, cmd[:80]))
for pid, cmd in victims:
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"killed {pid}: {cmd}", flush=True)
    except ProcessLookupError:
        pass
time.sleep(5)
for pid, cmd, env in procs():
    if KILL_PAT and "depth_eval.py" in cmd and KILL_PAT in cmd and f"CUDA_VISIBLE_DEVICES={GPU}" in env:
        os.kill(pid, signal.SIGKILL)
        print(f"force-killed {pid}", flush=True)
with open(REPORT, "a") as f:
    f.write(f"{NOTE} {time.strftime('%m-%d %H:%M')}\n")
print("done", flush=True)
