#!/bin/bash
#SBATCH --job-name=ouro_he_sanity
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40g
#SBATCH --time=01:30:00
#SBATCH --output=logs/ouro_he_sanity_%j.out
#SBATCH --error=logs/ouro_he_sanity_%j.err

# Sanity check: evaluate the RAW pretrained ByteDance/Ouro-1.4B on HumanEval,
# loading weights directly from HF safetensors (no SFT, no DCP checkpoint).
# Purpose: confirm the base model produces real code and a non-zero pass@1,
# isolating the "pass@1 = 0" bug to the SFT-from-random-init path (now fixed
# in ouro_1_4b_sft via checkpoint.initial_load_path).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

export TMPDIR=${TMPDIR:-/scratch/bcjw/jli37/tmp}
mkdir -p "$TMPDIR"

source .venv/bin/activate

# Install lm-evaluation-harness with code-eval support if not present
python -c "import lm_eval" 2>/dev/null || pip install --quiet 'lm_eval[code]'

export MODULE=${MODULE:-ouro}
export CONFIG=${CONFIG:-ouro_1_4b}
export HF_CHECKPOINT=${HF_CHECKPOINT:-./assets/hf/Ouro-1.4B}
export LIMIT=${LIMIT:-20}           # leave empty to run all 164 problems
export MAX_GEN_TOKS=${MAX_GEN_TOKS:-512}
export ALLOW_CODE_EXECUTION=1
export HF_ALLOW_CODE_EVAL=1
# CHAT=1 uses the ChatML chat template (SFT-matched); Ouro-1.4B is chat-tuned,
# so the raw-completion protocol scores ~0. Default on for this model.
export CHAT=${CHAT:-1}

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi
CHAT_ARG=""
if [ "$CHAT" = "1" ]; then
    CHAT_ARG="--chat"
fi

# No --early_exit_threshold: the base model runs its native UT recursion
# (threshold defaults to 1.0 => full recursion, no early exit).
python -u scripts/evaluate_humaneval.py \
    --module "$MODULE" \
    --config "$CONFIG" \
    --hf_checkpoint "$HF_CHECKPOINT" \
    --max_gen_toks "$MAX_GEN_TOKS" \
    --output_path "outputs/humaneval_hf_sanity_${SLURM_JOB_ID}.json" \
    $LIMIT_ARG \
    $CHAT_ARG \
    "$@"
