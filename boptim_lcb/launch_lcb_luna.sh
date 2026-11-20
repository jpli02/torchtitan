#!/bin/bash
# LLM-as-optimizer (openai/gpt-5.6-luna through OpenRouter) on the pooled LiveCodeBench router objective: all 342 problems,
# 10 new evaluations after the same 7 warm-start records the Gaussian-process run had (user request 2026-11-20).
# One evaluation at a time on one card (about 3 hours each: generation + contention-checked speed calibration).
#   GPU=1 bash boptim_lcb/launch_lcb_luna.sh        start detached with setsid nohup; log: <job tmp>/lcb_luna.log
# The API key is read from ~/.boptim_keys.env and never echoed.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
WORK=/home/jli199/boptim_scratch/bo_lcb_router_pooled_luna
GPU=${GPU:-1}
set -a
. "$HOME/.boptim_keys.env"
set +a
KEY=${OPENROUTER_API_KEY:-${QWEN_API_KEY:-}}
[ -n "$KEY" ] || { echo "no OpenRouter key in ~/.boptim_keys.env"; exit 1; }
export OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY="$KEY" GPT_MAX_TOKENS=16000
export BOPTIM_PROMPT_TEMPLATE=ouro_lcb_router_prompt.txt
cd /home/jli199/boptim-agent || exit 1
cp "$WT/boptim_lcb/ouro_lcb_router_prompt.txt" optim/ouro_lcb_router_prompt.txt
cp "$WT/boptim_lcb/ouro_lcb_router_pooled_luna.yaml" configs/ouro_lcb_router_pooled_luna.yaml
mkdir -p "$WORK"
: > "$WORK/gpus.txt"
./.venv/bin/python "$WT/boptim_lcb/make_warm_start_luna.py" 2>&1 | grep -v Warning
echo "LLM search starts on gpu $GPU $(date '+%m-%d %H:%M')"
LCB_BO_GPUS=$GPU ./.venv/bin/python -u run_experiment.py --config configs/ouro_lcb_router_pooled_luna.yaml --log-prefix lcb_router_pooled_luna \
  > "$TMP/lcb_luna.log" 2>&1
echo "LUNA search exited $? $(date '+%m-%d %H:%M')"
