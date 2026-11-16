"""Ask this user's processes whose command line contains PATTERN to stop with SIGINT (Ctrl-C) and wait for them.

Unlike kill_by_pattern.py nothing is killed: a Terminal-Bench harness stopped this way runs its own cleanup and takes its
trial container down, so no left-over container is reused by a later run. Python 3.6 compatible.
  python3 -I stop_gracefully.py PATTERN [WAIT_SECONDS]
"""
import os
import signal
import sys
import time

pat = sys.argv[1]
wait = int(sys.argv[2]) if len(sys.argv) > 2 else 120
me = {os.getpid(), os.getppid()}
victims = []
for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) in me:
        continue
    try:
        cmd = open("/proc/%s/cmdline" % pid, "rb").read().replace(b"\0", b" ").decode(errors="replace")
        uid = os.stat("/proc/%s" % pid).st_uid
    except Exception:
        continue
    if uid == os.getuid() and pat in cmd and "stop_gracefully.py" not in cmd:
        victims.append((int(pid), cmd[:90]))
for pid, cmd in victims:
    try:
        os.kill(pid, signal.SIGINT)
        print("asked %d to stop: %s" % (pid, cmd))
    except ProcessLookupError:
        pass
end = time.time() + wait
alive = [p for p, _ in victims]
while alive and time.time() < end:
    time.sleep(3)
    still = []
    for p in alive:
        try:
            os.kill(p, 0)
            still.append(p)
        except ProcessLookupError:
            pass
    alive = still
print("%d asked, %d still running after %d s" % (len(victims), len(alive), wait))
