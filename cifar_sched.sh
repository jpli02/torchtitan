#!/bin/bash
# Scheduler: keep one gen_bo_traj process per GPU (8 GPUs), each doing exactly
# ONE trajectory in a fresh process (avoids the CUDA context-reuse error that
# killed the 2nd trajectory in a reused process). Refill a GPU slot as soon as
# its process exits; stop when total observations across saved JSONs >= TARGET.
set -u
REPO=/home/jli199/boptim-agent
PY=$REPO/.venv/bin/python
OUT=/home/jli199/boptim_scratch/cifar_gpt_1k
WORK=/home/jli199/boptim_scratch/cifar_1k_work
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PROG=$TMP/cifar_1k_progress.txt
GPUS="1 2 3 4 5 6 7 8"
TARGET=${TARGET:-1000}
SEED=${SEED_START:-2210}
MAXIT=${MAXIT:-15}
mkdir -p "$OUT"

count_obs(){ "$PY" - "$OUT" <<'PYEOF'
import sys, json, glob, os
d = sys.argv[1]; o = 0
for f in glob.glob(os.path.join(d, "*.json")):
    try: o += len(json.load(open(f)).get("observations", []))
    except Exception: pass
print(o)
PYEOF
}

LAST_PID=0
launch(){ # $1=gpu $2=seed -> sets LAST_PID (direct bg launch, NO command subst)
  local g=$1 s=$2 wd="$WORK/g$g"
  mkdir -p "$wd"; cd "$wd"
  CUDA_VISIBLE_DEVICES=$g BOPTIM_REPO=$REPO nohup "$PY" "$REPO/gen_bo_traj.py" \
    --objective cifar --direction maximize --optim chatgpt --model gpt-5.6-luna \
    --n 1 --seed_start "$s" --max_iter "$MAXIT" --epochs 10 --out "$OUT" \
    > "$TMP/cifar_g${g}_s${s}.log" 2>&1 &
  LAST_PID=$!
}

declare -A PID
echo "[cifar_sched] start $(date '+%m-%d %H:%M') obs=$(count_obs) target=$TARGET seed0=$SEED" >> "$PROG"
for g in $GPUS; do launch "$g" "$SEED"; PID[$g]=$LAST_PID; SEED=$((SEED+1)); done

tick=0
END=$(( $(date +%s) + 79200 ))   # 22h safety cap
while [ "$(date +%s)" -lt "$END" ]; do
  sleep 120; tick=$((tick+1))
  obs=$(count_obs)
  if [ "$obs" -ge "$TARGET" ]; then
    echo "[cifar_sched] TARGET reached obs=$obs $(date '+%m-%d %H:%M'); stopping" >> "$PROG"
    for g in $GPUS; do kill -9 "${PID[$g]}" 2>/dev/null; done
    pkill -9 -f gen_bo_traj.py 2>/dev/null
    break
  fi
  for g in $GPUS; do
    if ! kill -0 "${PID[$g]}" 2>/dev/null; then
      launch "$g" "$SEED"; PID[$g]=$LAST_PID; SEED=$((SEED+1))
    fi
  done
  if [ $((tick % 5)) -eq 0 ]; then
    njson=$(ls "$OUT"/*.json 2>/dev/null | wc -l)
    echo "$(date '+%m-%d %H:%M') sched trajectories=$njson obs=$obs next_seed=$SEED running=$(pgrep -f gen_bo_traj.py | grep -cve '^$')" >> "$PROG"
  fi
done
echo "[cifar_sched] done $(date '+%m-%d %H:%M') obs=$(count_obs) trajectories=$(ls "$OUT"/*.json 2>/dev/null | wc -l)" >> "$PROG"
