#!/bin/bash
# Router search on the teacher-forced proxy objective for one system: baseline + warm start (tf_bo_prep.py), then the
# optimizer (boptim-agent run_experiment.py). One process per card; about a minute per evaluation.
#   GPU=9 NAME=fill24 bash randrouter/tf_bo.sh [tf_bo_prep.py options, e.g. --fill_maps F --lora L --gate G --new 50 --mem_gb 8.3]
# Log: /home/jli199/boptim_scratch/randrouter/bo_tf/<NAME>/bo.log ; results: evaluations.jsonl in the same directory.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
BO=/home/jli199/boptim-agent
NAME=${NAME:?set NAME}
GPU=${GPU:-0}
W=/home/jli199/boptim_scratch/randrouter/bo_tf/$NAME
mkdir -p "$W"
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True
"$BO/.venv/bin/python" "$WT/randrouter/setup_tf_objective.py" > "$W/setup.log" 2>&1
cd "$BO" || exit 1
./.venv/bin/python -u "$WT/randrouter/tf_bo_prep.py" --name "$NAME" "$@" > "$W/prep.log" 2>&1 || { echo "prep failed, see $W/prep.log"; exit 1; }
if [ -n "${LLM_KEY_ENV:-}" ]; then   # an LLM optimizer: key from ~/.boptim_keys.env, never echoed
  set -a; . "$HOME/.boptim_keys.env"; set +a
  KEY=${OPENROUTER_API_KEY:-${QWEN_API_KEY:-}}
  export OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY="$KEY" GPT_MAX_TOKENS=16000
fi
./.venv/bin/python -u run_experiment.py --config "configs/ouro_tf_$NAME.yaml" --log-prefix "tf_$NAME" > "$W/bo.log" 2>&1
echo "TF search $NAME exited $? $(date '+%m-%d %H:%M')"
