#!/bin/bash
#SBATCH --job-name=he_evaltest
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=eval_only_test_%j.out
#SBATCH --error=eval_only_test_%j.err

# Eval-only smoke test: run HumanEval directly on an existing checkpoint
# (no training) to verify the eval pipeline is correct after the torchvision /
# code_eval / unsafe-task / pass@1-parse fixes.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

CKPT="outputs/boptim/eval000_lr3.16e-04_20260612_170547/checkpoint/step-5"
OUT="outputs/boptim/eval000_lr3.16e-04_20260612_170547/humaneval_evaltest.json"

.venv/bin/python scripts/evaluate_humaneval.py \
    --module ouro --config ouro_1_4b \
    --checkpoint "$CKPT" \
    --max_gen_toks 256 \
    --early_exit_threshold 0.9 \
    --output_path "$OUT" \
    --limit 20
