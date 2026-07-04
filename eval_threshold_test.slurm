#!/bin/bash
#SBATCH --job-name=he_thr
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:20:00
#SBATCH --output=eval_thr_%j.out
#SBATCH --error=eval_thr_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false
CKPT="outputs/boptim/eval000_lr3.16e-04_20260612_170547/checkpoint/step-5"
# Full UT stack (threshold=1.0, gate unused) -- isolates whether early exit is the culprit.
.venv/bin/python scripts/evaluate_humaneval.py \
    --module ouro --config ouro_1_4b \
    --checkpoint "$CKPT" \
    --max_gen_toks 256 \
    --early_exit_threshold 1.0 \
    --output_path outputs/boptim/eval000_lr3.16e-04_20260612_170547/humaneval_thr1.0.json \
    --limit 5
