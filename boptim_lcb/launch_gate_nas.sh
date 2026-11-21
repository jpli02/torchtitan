#!/bin/bash
# Gate NAS on LiveCodeBench with real execution (objective ouro_lcb_gate_nas, boptim_lcb/ouro_lcb_gate_nas.py): the HumanEval
# router-NAS experiment on Ouro-2.6B (user, 2026-11-21: "only train the gate, like HumanEval"). Same objective, budget and seed
# rules for every optimizer; no warm start (the gates are trained from scratch, so nothing earlier is in this space).
#   OPTIM=gp   GPU=0 bash boptim_lcb/launch_gate_nas.sh                      Gaussian process (skopt gp_hedge, 5 random points first)
#   OPTIM=luna GPU=1 PROVIDER=openai bash boptim_lcb/launch_gate_nas.sh      LLM agent gpt-6-luna (OpenAI API, key file's OPENAI_API_KEY)
#   OPTIM=random GPU=... bash boptim_lcb/launch_gate_nas.sh                 uniform random search
# ITERS (default 25) = evaluations; SEED (default 1). Results: /home/jli199/boptim_scratch/bo_lcb_gate_nas_<optim>_s<seed>/
# (evaluations.jsonl, routers/, one run dir per decoded evaluation); log in the job tmp gate_nas_<optim>_s<seed>.log.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
BO=/home/jli199/boptim-agent
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
OPTIM=${OPTIM:-gp}
GPU=${GPU:-0}
ITERS=${ITERS:-25}
SEED=${SEED:-1}
NAME=${OPTIM}_s${SEED}
WORK=/home/jli199/boptim_scratch/bo_lcb_gate_nas_${NAME}
mkdir -p "$WORK/runs"
"$BO/.venv/bin/python" "$WT/boptim_lcb/setup_gate_nas.py" > /dev/null || { echo "setup failed"; exit 1; }
cd "$BO" || exit 1
CFG=configs/ouro_lcb_gate_nas_${NAME}.yaml
MODEL_LINE=""
case "$OPTIM" in
  gp) OPT=gp_hedge ;;
  random) OPT=random ;;
  luna)
    OPT=chatgpt
    set -a; . "$HOME/.boptim_keys.env"; set +a
    if [ "${PROVIDER:-openai}" = openai ]; then
      MODEL=${MODEL:-gpt-6-luna}
      export OPENAI_BASE_URL=https://api.openai.com/v1
    else
      MODEL=${MODEL:-openai/gpt-6-luna}
      export OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY="${OPENROUTER_API_KEY:-}"
    fi
    export GPT_MAX_TOKENS=16000
    MODEL_LINE="models:
  chatgpt: \"$MODEL\""
    TPL=ouro_lcb_gate_nas_${ITERS}_prompt.txt
    sed "s/__ITERS__/$ITERS/g" "$WT/randrouter/prompts/lcb_gate_nas_informed.txt" > "optim/$TPL"
    export BOPTIM_PROMPT_TEMPLATE=$TPL
    PING_MODEL="$MODEL" ./.venv/bin/python "$WT/randrouter/llm_ping.py" || { echo "LLM ping failed, not started"; exit 1; }
    ;;
  *) echo "OPTIM must be gp, random or luna"; exit 1 ;;
esac
cat > "$CFG" <<EOF
# gate NAS on LiveCodeBench, real execution (boptim_lcb/launch_gate_nas.sh): optimizer $OPTIM, seed $SEED
replicates: 1
no_explanation: false
wandb:
  enabled: false
$MODEL_LINE
experiments:
  - optim: $OPT
    objective:
      type: ouro_lcb_gate_nas
      ouro_lcb_n: 342
      ouro_lcb_baseline_acc: 20.5
      ouro_lcb_tol: 2.2
      ouro_lcb_penalty: 2.0
      ouro_lcb_work_dir: $WORK
    batch_size: 1
    max_iter: $ITERS
    resume: true
    resume_dir: $WORK/runs
    optim_seed: $SEED
    objective_seed: 0
    n_initial: 5
EOF
echo "gate NAS ($OPTIM, seed $SEED) starts on gpu $GPU $(date '+%m-%d %H:%M')"
LCB_BO_GPUS=$GPU ./.venv/bin/python -u run_experiment.py --config "$CFG" --log-prefix "lcb_gate_nas_${NAME}" > "$TMP/gate_nas_${NAME}.log" 2>&1
echo "gate NAS ($OPTIM, seed $SEED) exited $? $(date '+%m-%d %H:%M')"
