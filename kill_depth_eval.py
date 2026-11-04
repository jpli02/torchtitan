"""Kill depth_eval.py processes whose command line contains PATTERN and whose CUDA_VISIBLE_DEVICES equals GPU
(scans /proc, so the caller's own shell never matches). Usage: python kill_depth_eval.py PATTERN GPU"""
import os
import signal
import sys
import time

pat, gpu = sys.argv[1], sys.argv[2]
victims = []
for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) == os.getpid():
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        env = open(f"/proc/{pid}/environ", "rb").read().replace(b"\0", b"\n").decode(errors="replace").split("\n")
    except Exception:
        continue
    if "depth_eval.py" in cmd and pat in cmd and f"CUDA_VISIBLE_DEVICES={gpu}" in env:
        victims.append(int(pid))
for p in victims:
    os.kill(p, signal.SIGTERM)
    print(f"terminated {p}")
time.sleep(5)
for p in victims:
    try:
        os.kill(p, 0)
        os.kill(p, signal.SIGKILL)
        print(f"killed {p}")
    except ProcessLookupError:
        pass
print(f"{len(victims)} process(es) matched")
