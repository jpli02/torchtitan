#!/bin/bash
# Watch the cifar_gpt_1k output; when total observations across completed
# trajectory JSONs reaches TARGET, stop all workers. Logs progress each tick.
set -u
OUT=/home/jli199/boptim_scratch/cifar_gpt_1k
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
TARGET=${TARGET:-1000}
PROG=$TMP/cifar_1k_progress.txt
END=$(( $(date +%s) + 54000 ))   # 15h safety cap
while [ "$(date +%s)" -lt "$END" ]; do
  read TRAJ OBS <<< "$($PY - "$OUT" <<'PYEOF'
import sys, json, glob, os
d = sys.argv[1]; traj = 0; obs = 0
for f in glob.glob(os.path.join(d, "*.json")):
    try:
        j = json.load(open(f)); traj += 1; obs += len(j.get("observations", []))
    except Exception:
        pass
print(traj, obs)
PYEOF
)"
  workers=$(pgrep -f "gen_bo_traj.py" | grep -cve '^$')
  echo "$(date '+%m-%d %H:%M')  trajectories=$TRAJ  observations=$OBS  workers=$workers" >> "$PROG"
  if [ "${OBS:-0}" -ge "$TARGET" ]; then
    echo "$(date '+%m-%d %H:%M')  TARGET $TARGET reached (obs=$OBS); stopping workers" >> "$PROG"
    pkill -9 -f "gen_bo_traj.py" 2>/dev/null
    break
  fi
  if [ "${workers:-0}" -eq 0 ] && [ "${OBS:-0}" -gt 0 ]; then
    echo "$(date '+%m-%d %H:%M')  all workers exited at obs=$OBS (below target)" >> "$PROG"
    break
  fi
  sleep 600
done
echo "$(date '+%m-%d %H:%M')  monitor done: trajectories=$TRAJ observations=$OBS" >> "$PROG"
