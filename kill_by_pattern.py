"""Kill every process of this user whose command line contains PATTERN (scans /proc; never matches its own shell).
Usage: python kill_by_pattern.py PATTERN [PATTERN2 ...]"""
import os
import signal
import sys
import time

pats = sys.argv[1:]
me = {os.getpid(), os.getppid()}
victims = []
for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) in me:
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        uid = os.stat(f"/proc/{pid}").st_uid
    except Exception:
        continue
    if uid != os.getuid() or "kill_by_pattern.py" in cmd:
        continue
    if any(p in cmd for p in pats):
        victims.append((int(pid), cmd[:90]))
for pid, cmd in victims:
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"terminated {pid}: {cmd}")
    except ProcessLookupError:
        pass
time.sleep(5)
for pid, _ in victims:
    try:
        os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL)
        print(f"killed {pid}")
    except ProcessLookupError:
        pass
print(f"{len(victims)} matched")
