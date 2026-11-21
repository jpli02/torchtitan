#!/bin/bash
# Optimizer against optimizer on the fast router-NAS objective (ouro_tf_nas: router architecture + Stage-II gamma + exit
# threshold, every candidate trained from a random initialisation; randrouter/ouro_tf_nas.py). All arms share the objective,
# the budget and the result layout, so randrouter/opt_compare_report.py reads them side by side.
#   GPU=9 ARMS="random gp_hedge luna-blind luna-informed" SEEDS="1 2 3 4 5" ITERS=40 bash randrouter/nas_compare.sh
#   random, gp_hedge   uniform random search; scikit-optimize Gaussian process (N_INITIAL random points first, default 8)
#   <label>-blind      LLM agent with the generic prompt: bounds and the history of (point, value) only
#   <label>-informed   LLM agent with randrouter/prompts/tf_<TAG>_informed.txt: what every dimension is and what is measured
#   label -> model: luna = openai/gpt-6-luna (override with LLM_<LABEL>=<model id>, e.g. LLM_sol=openai/gpt-6-sol)
# TAG names the space (result directory and prompt file); nas6 = the 6-dimensional space of objective/ouro_router_nas.py,
# nas8 = three per-loop thresholds instead of one (PER_LOOP=1 is set from the tag). An LLM agent has no seed: its "seeds" are
# independent repeats. Results: <ROOT>/<TAG>/<arm>_s<seed>/evaluations.jsonl; LLM prompts and answers under boptim-agent/logs/.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
BO=/home/jli199/boptim-agent
ROOT=/home/jli199/boptim_scratch/randrouter/opt_compare
GPU=${GPU:-9}
TAG=${TAG:-nas6}
ARMS=${ARMS:-random gp_hedge}
SEEDS=${SEEDS:-1 2 3 4 5}
ITERS=${ITERS:-40}
N_INITIAL=${N_INITIAL:-8}
MEM=${MEM_GB:-8.5}
PER_LOOP=0; [ "$TAG" = nas8 ] && PER_LOOP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [ -n "${WAIT_PID:-}" ]; then until ! kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; fi
"$BO/.venv/bin/python" "$WT/randrouter/setup_tfnas_objective.py" > /dev/null 2>&1
cd "$BO" || exit 1
NEED_LLM=0
for arm in $ARMS; do case "$arm" in *-blind|*-informed) NEED_LLM=1;; esac; done
if [ "$NEED_LLM" = 1 ]; then
  set -a
  . "$HOME/.boptim_keys.env"
  set +a
  KEY=${OPENROUTER_API_KEY:-${QWEN_API_KEY:-}}
  [ -n "$KEY" ] || { echo "no OpenRouter key in ~/.boptim_keys.env"; exit 1; }
  export OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY="$KEY" GPT_MAX_TOKENS=16000
fi
for seed in $SEEDS; do
  for arm in $ARMS; do
    W=$ROOT/$TAG/${arm}_s$seed
    if [ "$(cat "$W/evaluations.jsonl" 2>/dev/null | wc -l)" -ge "$ITERS" ]; then continue; fi
    OPTIM=$arm; MODEL=""; TPL=""
    case "$arm" in
      *-blind|*-informed)
        LABEL=${arm%-*}; KIND=${arm##*-}
        VAR=LLM_$LABEL; MODEL=${!VAR:-}
        if [ -z "$MODEL" ]; then [ "$LABEL" = luna ] && MODEL=openai/gpt-6-luna || { echo "no model for label $LABEL (set LLM_$LABEL)"; continue; }; fi
        OPTIM=chatgpt
        if [ "$KIND" = blind ]; then
          TPL=chatgpt_prompt.txt
        else
          SRC=$WT/randrouter/prompts/tf_${TAG}_informed.txt
          [ -f "$SRC" ] || { echo "no prompt file $SRC, arm $arm skipped"; continue; }
          TPL=ouro_tf_${TAG}_informed_${ITERS}_prompt.txt
          sed "s/__ITERS__/$ITERS/g" "$SRC" > "optim/$TPL"
        fi
        PING_MODEL="$MODEL" ./.venv/bin/python "$WT/randrouter/llm_ping.py" > /dev/null || { echo "$(date '+%m-%d %H:%M') LLM ping failed for $MODEL, $arm seed $seed skipped"; continue; }
        ;;
    esac
    rm -rf "$W"; mkdir -p "$W/runs"
    CFG=configs/ouro_tf_cmp_${TAG}_${arm}_s$seed.yaml
    {
      echo "# optimizer comparison on the fast router-NAS objective (randrouter/nas_compare.sh): space $TAG, arm $arm, seed $seed"
      echo "replicates: 1"
      echo "no_explanation: false"
      echo "wandb:"
      echo "  enabled: false"
      if [ -n "$MODEL" ]; then echo "models:"; echo "  chatgpt: \"$MODEL\""; fi
      echo "experiments:"
      echo "  - optim: $OPTIM"
      echo "    objective:"
      echo "      type: ouro_tf_nas"
      echo "      ouro_tfnas_n: 99"
      echo "      ouro_tfnas_floor: 98.45"
      echo "      ouro_tfnas_penalty: 3.5"
      echo "      ouro_tfnas_mem_gb: $MEM"
      echo "      ouro_tfnas_work_dir: $W"
      echo "      ouro_tfnas_per_loop: $PER_LOOP"
      echo "    batch_size: 1"
      echo "    max_iter: $ITERS"
      echo "    resume: false"
      echo "    resume_dir: $W/runs"
      echo "    optim_seed: $seed"
      echo "    objective_seed: 0"
      if [ "$OPTIM" = gp_hedge ]; then echo "    n_initial: $N_INITIAL"; fi
    } > "$CFG"
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -n 1)" -ge 8800 ]; do sleep 30; done
    [ -n "$TPL" ] && echo "$TPL $MODEL" > "$W/prompt_template.txt"
    BOPTIM_PROMPT_TEMPLATE=${TPL:-chatgpt_prompt.txt} CUDA_VISIBLE_DEVICES=$GPU ./.venv/bin/python -u run_experiment.py --config "$CFG" --log-prefix "cmp_${TAG}_${arm}_s$seed" > "$W/bo.log" 2>&1
    echo "$(date '+%m-%d %H:%M') $TAG $arm seed $seed: $(cat "$W/evaluations.jsonl" 2>/dev/null | wc -l) evaluations"
  done
done
echo "NAS COMPARE $TAG done $(date '+%m-%d %H:%M')"
