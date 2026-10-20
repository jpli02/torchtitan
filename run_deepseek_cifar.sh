#!/bin/bash
# Route the ChatGPT optimizer (openai SDK) at OpenRouter for the deepseek
# teacher: openai.OpenAI() honors OPENAI_BASE_URL / OPENAI_API_KEY.
set -u
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
exec env \
  GPUS="6 8" \
  OUT=/home/jli199/boptim_scratch/cifar_deepseek41flash \
  MODEL="deepseek/deepseek-v4.1-flash" \
  MAXIT=10 METRIC=traj TARGET=100 SEED_START=50000 \
  GPT_REASONING_EFFORT=high GPT_MAX_TOKENS=40000 \
  bash /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/cifar_sched.sh
