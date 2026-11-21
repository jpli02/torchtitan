#!/bin/bash
# Optimizer against optimizer on the teacher-forced router proxy (ouro_tf_router), several seeds each, no warm start: the test
# of "can this landscape tell optimizers apart" that has to precede any claim about an LLM agent's optimization ability
# (paper goal stated by the user on 2026-11-20: show that an LLM agent can optimize; the scenario is NAS).
#   GPU=9 SPACE=thr   OPTIMS="random gp_hedge" SEEDS="1 2 3 4 5 6 7 8 9 10" ITERS=30 bash randrouter/opt_compare.sh
#   SPACE: thr   = three exit thresholds, released model, decoder's copy fill (the space of the real searches)
#          exact = the same thresholds with an exact cache (the proxy's optimistic upper bound; a harder, more structured landscape)
#          veto  = thresholds + loop-1 veto threshold with the calibrated fill (four variables)
#   An LLM optimizer: OPTIMS="chatgpt" MODEL=openai/gpt-6-luna (key from ~/.boptim_keys.env, never echoed).
# One run = 30 evaluations of about 15 s on an idle card. Results: <ROOT>/<space>/<optim>_s<seed>/evaluations.jsonl
# (report: randrouter/opt_compare_report.py). A finished run is skipped, so the script can be restarted.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
BO=/home/jli199/boptim-agent
ROOT=/home/jli199/boptim_scratch/randrouter/opt_compare
GPU=${GPU:-9}
SPACE=${SPACE:-thr}
OPTIMS=${OPTIMS:-random gp_hedge}
SEEDS=${SEEDS:-1 2 3 4 5 6 7 8 9 10}
ITERS=${ITERS:-30}
MODEL=${MODEL:-openai/gpt-6-luna}
MEM=${MEM_GB:-8.5}
FILL=/home/jli199/boptim_scratch/randrouter/fill_maps_lcb_n24.pt
PROBE=/home/jli199/boptim_scratch/randrouter/harm_probe_fill24/linear.pt
export PYTORCH_ALLOC_CONF=expandable_segments:True
# WAIT_PID: start only after that process has ended (a job of mine that reloads the model on this card between its chunks,
# so "the card has room right now" is not enough to take it)
if [ -n "${WAIT_PID:-}" ]; then until ! kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; fi
"$BO/.venv/bin/python" "$WT/randrouter/setup_tf_objective.py" > /dev/null 2>&1
cd "$BO" || exit 1
for seed in $SEEDS; do
  for optim in $OPTIMS; do
    W=$ROOT/$SPACE/${optim}_s$seed
    if [ "$(cat "$W/evaluations.jsonl" 2>/dev/null | wc -l)" -ge "$ITERS" ]; then continue; fi
    rm -rf "$W"; mkdir -p "$W/runs"
    CFG=configs/ouro_tf_cmp_${SPACE}_${optim}_s$seed.yaml
    {
      echo "# optimizer comparison on the router proxy (randrouter/opt_compare.sh): space $SPACE, optimizer $optim, seed $seed"
      echo "replicates: 1"
      echo "wandb:"
      echo "  enabled: false"
      if [ "$optim" = chatgpt ]; then echo "models:"; echo "  chatgpt: \"$MODEL\""; fi
      echo "experiments:"
      echo "  - optim: $optim"
      echo "    objective:"
      echo "      type: ouro_tf_router"
      echo "      ouro_tf_n: 99"
      echo "      ouro_tf_floor: 98.45"
      echo "      ouro_tf_penalty: 3.5"
      echo "      ouro_tf_mem_gb: $MEM"
      echo "      ouro_tf_work_dir: $W"
      if [ "$SPACE" = exact ]; then echo "      ouro_tf_exact: 1"; fi
      if [ "$SPACE" = veto ]; then echo "      ouro_tf_fill_maps: $FILL"; echo "      ouro_tf_veto: $PROBE"; fi
      echo "    batch_size: 1"
      echo "    max_iter: $ITERS"
      echo "    resume: false"
      echo "    resume_dir: $W/runs"
      echo "    optim_seed: $seed"
      echo "    objective_seed: 0"
      if [ "$optim" = gp_hedge ]; then echo "    n_initial: 5"; fi
    } > "$CFG"
    # wait until the card has room for the model (another job of mine, or another user's, may hold it)
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -n 1)" -ge 8800 ]; do sleep 30; done
    if [ "$optim" = chatgpt ]; then
      set -a; . "$HOME/.boptim_keys.env"; set +a
      export OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY="${OPENROUTER_API_KEY:-${QWEN_API_KEY:-}}" GPT_MAX_TOKENS=16000
    fi
    CUDA_VISIBLE_DEVICES=$GPU ./.venv/bin/python -u run_experiment.py --config "$CFG" --log-prefix "cmp_${SPACE}_${optim}_s$seed" > "$W/bo.log" 2>&1
    echo "$(date '+%m-%d %H:%M') $SPACE $optim seed $seed: $(cat "$W/evaluations.jsonl" 2>/dev/null | wc -l) evaluations"
  done
done
echo "OPT COMPARE $SPACE done $(date '+%m-%d %H:%M')"
