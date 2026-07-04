#!/bin/bash
#SBATCH --job-name=ouro_parity
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=00:30:00
#SBATCH --output=logs/ouro_parity_%j.out
#SBATCH --error=logs/ouro_parity_%j.err

# Logit parity: HF modeling_ouro.py vs torchtitan OuroModel (same weights, R=4).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate
python -u scripts/parity_hf_vs_tt.py
