#!/bin/bash
# Key check for the LLM-optimizer runs: remaining credit (numbers only) and one small call. Prints no key material.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
set -a
. "$HOME/.boptim_keys.env"
set +a
export OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY="${OPENROUTER_API_KEY:-${QWEN_API_KEY:-}}"
cd /home/jli199/boptim-agent || exit 1
./.venv/bin/python "$WT/randrouter/llm_credit.py"
PING_MODEL="${MODEL:-openai/gpt-6-luna}" ./.venv/bin/python "$WT/randrouter/llm_ping.py"
