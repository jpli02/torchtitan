#!/bin/bash
# Router-architecture NAS through BO: gp_hedge vs an LLM agent optimizer.
#
# Space (6-D): 4 discrete architecture axes for Ouro's early-exit gate
#   layers in {0,1,2}, hidden in {64,128,256}, act in {relu,gelu,silu,tanh},
#   norm in {False,True}
# plus the two continuous knobs (adaptive_gamma, early_exit_threshold), which
# are searched JOINTLY because the best operating point is architecture-specific
# -- fixing them would make architectures that want a different threshold look
# artificially bad.
#
# Objective (existing convention): MINIMIZE avg_loops subject to an accuracy
# floor, else return loop_penalty=10.0. That directly encodes "cheaper inference
# without losing ability" rather than scalarising the two.
#
# Each candidate router is trained from RANDOM INIT (pretrained gate weights
# cannot transfer to a different module shape) for --ouro_steps, then scored.
#
# Env: OPTIM (gp_hedge|claude|chatgpt|qwen), ITERS, GPU, TAG, MODEL
set -u
BO=/home/jli199/boptim-agent
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
: "${GPU:?GPU must be set explicitly}"
: "${OPTIM:?OPTIM must be set}"
ITERS=${ITERS:-1}
TAG=${TAG:-$OPTIM}
STEPS=${STEPS:-100}
LIMIT=${LIMIT:-20}

cd "$BO" || exit 1
if [ -f "$HOME/.boptim_keys.env" ]; then
  set -a
  . "$HOME/.boptim_keys.env"
  set +a
fi

EXTRA=""
[ -n "${MODEL:-}" ] && EXTRA="--model $MODEL"

echo "[nas] optim=$OPTIM iters=$ITERS gpu=$GPU steps=$STEPS limit=$LIMIT"
CUDA_VISIBLE_DEVICES=$GPU exec timeout 172800 .venv/bin/python main.py \
  --objective ouro_router_nas --optim "$OPTIM" --max_iter "$ITERS" \
  --torchtitan_dir /home/jli199/torchtitan \
  --ouro_ngpu 1 --ouro_eval_ngpu 1 \
  --ouro_steps "$STEPS" --ouro_limit "$LIMIT" \
  --ouro_ckpt_dir /home/jli199/boptim_scratch/nas_ckpt \
  $EXTRA
