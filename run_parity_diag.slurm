#!/bin/bash
#SBATCH --job-name=ouro_parity
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=00:25:00
#SBATCH --output=logs/ouro_parity_%j.out
#SBATCH --error=logs/ouro_parity_%j.err

# Run the torchtitan Ouro recurrence/early-exit diagnostic. Key signal:
#   GATHER early-exit tokens:  _gather_early_exits/_gather_total
# >0 at threshold=1.0 => the old threshold-gather scored logits from a PRE-FINAL
# UT step instead of the final R=4 step (the suspected bug degrading pass@1).
# Also reports best-effort HF-reference logit/argmax parity if it loads.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

python -u scripts/parity_hf_vs_tt.py
echo "=== parity diagnostic done ==="
