#!/bin/bash
# Fan out gpt-5.6-luna CIFAR BO trajectory generation across 8 GPUs, 2 jobs/GPU
# (16 workers). Each worker runs gen_bo_traj.py in its own cwd (isolated logs/),
# writing one JSON per trajectory to a shared OUT dir. Collects until a separate
# monitor sees >=1000 observations. Trajectories are durable as they complete.
set -u
REPO=/home/jli199/boptim-agent
PY=$REPO/.venv/bin/python
OUT=/home/jli199/boptim_scratch/cifar_gpt_1k
WORK=/home/jli199/boptim_scratch/cifar_1k_work
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
MODEL=${MODEL:-gpt-5.6-luna}
NPER=${NPER:-5}       # trajectories per worker (16*5=80 -> ~1200 obs, monitor caps at 1000)
MAXIT=${MAXIT:-15}
mkdir -p "$OUT"
echo "[cifar_1k] model=$MODEL workers=16 (gpu1-8 x2) n/worker=$NPER max_iter=$MAXIT $(date '+%m-%d %H:%M')" | tee "$TMP/cifar_1k_state.txt"

for w in $(seq 0 15); do
  gpu=$(( 1 + w / 2 ))
  seed=$(( 2000 + w * 10 ))
  wd="$WORK/w$w"
  mkdir -p "$wd"
  ( cd "$wd" && CUDA_VISIBLE_DEVICES=$gpu BOPTIM_REPO=$REPO nohup "$PY" "$REPO/gen_bo_traj.py" \
      --objective cifar --direction maximize --optim chatgpt --model "$MODEL" \
      --n "$NPER" --seed_start "$seed" --max_iter "$MAXIT" --epochs 10 \
      --out "$OUT" > "$TMP/cifar_w$w.log" 2>&1 & )
  echo "  worker $w -> gpu $gpu seeds $seed..$((seed+NPER-1))" | tee -a "$TMP/cifar_1k_state.txt"
  sleep 3
done
echo "[cifar_1k] all 16 workers launched $(date '+%m-%d %H:%M')" | tee -a "$TMP/cifar_1k_state.txt"
