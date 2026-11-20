#!/bin/bash
# Ping the LLM optimizer's model through OpenRouter with the key from ~/.boptim_keys.env (the key is never echoed).
#   bash boptim_lcb/ping_luna.sh [model]
set -a
. "$HOME/.boptim_keys.env"
set +a
KEY=${OPENROUTER_API_KEY:-${QWEN_API_KEY:-}}
[ -n "$KEY" ] || { echo "no OpenRouter key in ~/.boptim_keys.env"; exit 1; }
cd /home/jli199/boptim-agent || exit 1
OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY="$KEY" GPT_MAX_TOKENS=2000 \
  ./.venv/bin/python /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/boptim_lcb/ping_llm.py "${1:-openai/gpt-5.6-luna}"
