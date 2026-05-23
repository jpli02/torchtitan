#!/bin/bash
#SBATCH --job-name=ouro_humaneval
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40g
#SBATCH --time=01:30:00
#SBATCH --output=logs/ouro_humaneval_%j.out
#SBATCH --error=logs/ouro_humaneval_%j.err

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
export CHECKPOINT=${CHECKPOINT:-./outputs/checkpoint/step-500}
export LIMIT=${LIMIT:-10}          # leave empty to run all 164 problems
export MAX_GEN_TOKS=${MAX_GEN_TOKS:-512}
export ALLOW_CODE_EXECUTION=1

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

python -u scripts/evaluate_humaneval.py \
    --module "$MODULE" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --max_gen_toks "$MAX_GEN_TOKS" \
    --output_path "outputs/humaneval_${SLURM_JOB_ID}.json" \
    $LIMIT_ARG \
    "$@"
