#!/bin/bash
# Generate BO teacher trajectories WITH reasoning from a local Qwen teacher.
#
# Why local: the OpenRouter key is free-tier with ~3 cents left (402 "requires
# more credits" on every qwen3.7-plus call) and the OpenAI key has no credit,
# so the paid teachers are out. Qwen3-14B (bf16, 28GB) fits one free 45GB card;
# one $WT/research/qwen_bo_agent/qwen_bo_server.py per card, --no_think so each reply is just the JSON
# with analysis/plan (the reasoning we distil) instead of 1-2k think tokens.
#
#   TEACHER=.../Qwen3-14B GPUS="5 6 7 8" bash $WT/research/qwen_bo_agent/run_teacher_gen.sh
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
REPO=${BOPTIM_REPO:-$WT/research/boptim-agent}   # boptim-agent submodule (override with BOPTIM_REPO)
OUTROOT=${OUTROOT:-/home/jli199/boptim_scratch/bo_traj}
LOGDIR=/home/jli199/terminal_bench_eval/logs
TEACHER=${TEACHER:-/home/jli199/torchtitan/assets/hf/Qwen3-14B}
GPUS=${GPUS:-"5 6 7 8"}
BASE_PORT=${BASE_PORT:-8301}
MODEL_TAG=${MODEL_TAG:-qwen3-14b}
PER_WORKER_F2D=${PER_WORKER_F2D:-15}
PER_WORKER_RB=${PER_WORKER_RB:-3}
PER_WORKER_SS=${PER_WORKER_SS:-3}
MAX_ITER=${MAX_ITER:-15}
PY=/home/jli199/torchtitan/.venv/bin/python
RPY=$REPO/.venv/bin/python
mkdir -p "$OUTROOT"
cd "$WT" || exit 1

pids=()
i=0
for g in $GPUS; do
  port=$((BASE_PORT + i))
  CUDA_VISIBLE_DEVICES=$g "$PY" $WT/research/qwen_bo_agent/qwen_bo_server.py --hf_dir "$TEACHER" --port "$port" \
    --model_name "$MODEL_TAG" --no_think > "$LOGDIR/teacher_${MODEL_TAG}_gpu${g}.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
done
echo "started $i teacher servers (pids ${pids[*]}) $(date '+%m-%d %H:%M')"

# wait until every server answers /v1/models (model load ~1-2 min)
for _ in $(seq 1 120); do
  up=0
  for j in $(seq 0 $((i - 1))); do
    curl -s -m 3 "http://127.0.0.1:$((BASE_PORT + j))/v1/models" > /dev/null 2>&1 && up=$((up + 1))
  done
  [ "$up" = "$i" ] && break
  sleep 5
done
echo "servers up: $up/$i $(date '+%m-%d %H:%M')"

wpids=()
for j in $(seq 0 $((i - 1))); do
  url="http://127.0.0.1:$((BASE_PORT + j))/v1"
  (
    cd "$REPO" || exit 1
    "$RPY" $WT/research/qwen_bo_agent/gen_bo_traj.py --objective fourier2d --optim qwen --model "$MODEL_TAG" --qwen_base_url "$url" \
      --n "$PER_WORKER_F2D" --seed_start $((j * PER_WORKER_F2D)) --max_iter "$MAX_ITER" \
      --out "$OUTROOT/${MODEL_TAG}_fourier2d"
    "$RPY" $WT/research/qwen_bo_agent/gen_bo_traj.py --objective rosenbrock --optim qwen --model "$MODEL_TAG" --qwen_base_url "$url" \
      --n "$PER_WORKER_RB" --seed_start $((j * PER_WORKER_RB)) --max_iter "$MAX_ITER" \
      --out "$OUTROOT/${MODEL_TAG}_rosenbrock"
    "$RPY" $WT/research/qwen_bo_agent/gen_bo_traj.py --objective sumsq --optim qwen --model "$MODEL_TAG" --qwen_base_url "$url" \
      --n "$PER_WORKER_SS" --seed_start $((j * PER_WORKER_SS)) --max_iter "$MAX_ITER" \
      --out "$OUTROOT/${MODEL_TAG}_sumsq"
  ) > "$LOGDIR/teacher_gen_${MODEL_TAG}_w${j}.log" 2>&1 &
  wpids+=($!)
done
echo "started ${#wpids[@]} workers"
wait "${wpids[@]}"
echo "workers done $(date '+%m-%d %H:%M')"
kill "${pids[@]}" 2>/dev/null
for d in "$OUTROOT/${MODEL_TAG}"_*/; do printf "%s %s\n" "$(ls "$d" | wc -l)" "$d"; done
grep -h "FAILED" "$LOGDIR"/teacher_gen_${MODEL_TAG}_w*.log | cut -c1-160 | sort | uniq -c | head -5
echo "teacher gen finished"
