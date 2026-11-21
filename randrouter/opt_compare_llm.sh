#!/bin/bash
# LLM-agent arms of the optimizer comparison on the router proxy (same objective, budget and result layout as
# randrouter/opt_compare.sh, so randrouter/opt_compare_report.py reads them next to random search and the Gaussian process).
#   GPU=8 SPACE=thr ARMS="blind informed" REPS="1 2 3 4 5" ITERS=30 MODEL=openai/gpt-6-luna LABEL=luna bash randrouter/opt_compare_llm.sh
#   blind    = the generic prompt (optim/chatgpt_prompt.txt): bounds and the history of (point, value), nothing about the problem
#   informed = randrouter/prompts/tf_<space>_informed.txt: what the three thresholds do, what one evaluation measures and the
#              objective's formula (the mechanism text of the real LiveCodeBench search's prompt; nothing about where the optimum is)
# The LLM proposes every point from the first one (no random initial points) and has no seed: REPS are independent repeats.
# Calls go through OpenRouter with the key in ~/.boptim_keys.env (never echoed); gpt-6-luna costs 0.10 / 0.50 USD per million
# prompt / completion tokens, so a 30-evaluation run costs under one cent to a few cents.
# Results: <ROOT>/<space>/<label>-<arm>_s<rep>/evaluations.jsonl; every prompt and answer is kept under boptim-agent/logs/.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
BO=/home/jli199/boptim-agent
ROOT=/home/jli199/boptim_scratch/randrouter/opt_compare
GPU=${GPU:-8}
SPACE=${SPACE:-thr}
ARMS=${ARMS:-blind informed}
REPS=${REPS:-1 2 3 4 5}
ITERS=${ITERS:-30}
MODEL=${MODEL:-openai/gpt-6-luna}
LABEL=${LABEL:-luna}
MEM=${MEM_GB:-8.5}
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [ -n "${WAIT_PID:-}" ]; then until ! kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; fi
set -a
. "$HOME/.boptim_keys.env"
set +a
KEY=${OPENROUTER_API_KEY:-${QWEN_API_KEY:-}}
[ -n "$KEY" ] || { echo "no OpenRouter key in ~/.boptim_keys.env"; exit 1; }
export OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY="$KEY" GPT_MAX_TOKENS=16000
"$BO/.venv/bin/python" "$WT/randrouter/setup_tf_objective.py" > /dev/null 2>&1
cd "$BO" || exit 1
# one small call first: an expired key would otherwise fail every run after eight retries each
PING_MODEL="$MODEL" ./.venv/bin/python "$WT/randrouter/llm_ping.py" || { echo "LLM ping failed, nothing started $(date '+%m-%d %H:%M')"; exit 1; }
for rep in $REPS; do
  for arm in $ARMS; do
    NAME=${LABEL}-${arm}
    W=$ROOT/$SPACE/${NAME}_s$rep
    if [ "$(cat "$W/evaluations.jsonl" 2>/dev/null | wc -l)" -ge "$ITERS" ]; then continue; fi
    if [ "$arm" = blind ]; then
      TPL=chatgpt_prompt.txt
    else
      SRC=$WT/randrouter/prompts/tf_${SPACE}_${arm}.txt
      [ -f "$SRC" ] || { echo "no prompt file $SRC, arm skipped"; continue; }
      TPL=ouro_tf_${SPACE}_${arm}_prompt.txt
      cp "$SRC" "optim/$TPL"
    fi
    rm -rf "$W"; mkdir -p "$W/runs"
    CFG=configs/ouro_tf_cmp_${SPACE}_${NAME}_s$rep.yaml
    {
      echo "# optimizer comparison on the router proxy (randrouter/opt_compare_llm.sh): space $SPACE, LLM agent $MODEL, prompt $TPL, repeat $rep"
      echo "replicates: 1"
      echo "no_explanation: false"
      echo "wandb:"
      echo "  enabled: false"
      echo "models:"
      echo "  chatgpt: \"$MODEL\""
      echo "experiments:"
      echo "  - optim: chatgpt"
      echo "    objective:"
      echo "      type: ouro_tf_router"
      echo "      ouro_tf_n: 99"
      echo "      ouro_tf_floor: 98.45"
      echo "      ouro_tf_penalty: 3.5"
      echo "      ouro_tf_mem_gb: $MEM"
      echo "      ouro_tf_work_dir: $W"
      if [ "$SPACE" = exact ]; then echo "      ouro_tf_exact: 1"; fi
      echo "    batch_size: 1"
      echo "    max_iter: $ITERS"
      echo "    resume: false"
      echo "    resume_dir: $W/runs"
      echo "    optim_seed: $rep"
      echo "    objective_seed: 0"
    } > "$CFG"
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -n 1)" -ge 8800 ]; do sleep 30; done
    echo "$TPL" > "$W/prompt_template.txt"
    BOPTIM_PROMPT_TEMPLATE=$TPL CUDA_VISIBLE_DEVICES=$GPU ./.venv/bin/python -u run_experiment.py --config "$CFG" --log-prefix "cmp_${SPACE}_${NAME}_s$rep" > "$W/bo.log" 2>&1
    echo "$(date '+%m-%d %H:%M') $SPACE $NAME repeat $rep: $(cat "$W/evaluations.jsonl" 2>/dev/null | wc -l) evaluations"
  done
done
echo "OPT COMPARE LLM $SPACE done $(date '+%m-%d %H:%M')"
