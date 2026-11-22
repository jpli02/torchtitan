#!/bin/bash
# One arm x two seeds of the CIFAR-10 optimizer comparison (configs from cifar_bo/make_configs.py), detached so a daemon
# restart does not kill it:
#   ARM=luna6_default SEED=42 GPU=1 bash cifar_bo/launch.sh
# Runs from the boptim-agent worktree cifar-save-trajectory (its main.py saves trajectories); LLM arms call gpt-6-luna on
# OpenAI's own API with OPENAI_API_KEY from ~/.boptim_keys.env (never printed). Log: $LOGDIR/cifar_<ARM>_s<SEED>.log.
set -eu
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
BOPTIM=/home/jli199/boptim-agent/.claude/worktrees/cifar-save-trajectory
PY=/home/jli199/boptim-agent/.venv/bin/python
LOGDIR=${LOGDIR:-/home/jli199/boptim_scratch/cifar_bo_1122/logs}
CFG=$WT/cifar_bo/configs/${ARM}_s${SEED}.yaml
[ -f "$CFG" ] || { echo "no config $CFG"; exit 1; }
mkdir -p "$LOGDIR"

set -a
. "$HOME/.boptim_keys.env"
set +a
export OPENAI_BASE_URL=https://api.openai.com/v1 GPT_MAX_TOKENS=16000 CUDA_VISIBLE_DEVICES=$GPU
case "$ARM" in
  luna6_budget) export BOPTIM_PROMPT_TEMPLATE=$WT/cifar_bo/prompts/cifar_budget_blackbox.txt ;;
  luna6_informed) export BOPTIM_PROMPT_TEMPLATE=$WT/cifar_bo/prompts/cifar_informed.txt ;;
  luna6_minfix) export BOPTIM_PROMPT_TEMPLATE=$WT/cifar_bo/prompts/cifar_minfix.txt ;;
  luna6_minfix_high) export BOPTIM_PROMPT_TEMPLATE=$WT/cifar_bo/prompts/cifar_minfix.txt GPT_REASONING_EFFORT=high \
                       PYTHONPATH=$WT/cifar_bo/pyhooks ;;
  *) unset BOPTIM_PROMPT_TEMPLATE ;;
esac

cd "$BOPTIM"
echo "start $ARM seeds $SEED,$((SEED + 1)) on gpu $GPU $(date '+%m-%d %H:%M:%S') prompt=${BOPTIM_PROMPT_TEMPLATE:-default}" > "$LOGDIR/cifar_${ARM}_s${SEED}.log"
setsid nohup "$PY" -u run_experiment.py --config "$CFG" >> "$LOGDIR/cifar_${ARM}_s${SEED}.log" 2>&1 < /dev/null &
echo "launched $ARM s$SEED gpu $GPU pid $!"
