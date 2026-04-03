#!/bin/bash
#SBATCH --job-name=ouro_train
#SBATCH --account=bckr-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=02:30:00
#SBATCH --output=logs/ouro_train_%j.out
#SBATCH --error=logs/ouro_train_%j.err

# Ouro 1.4B training on GSM8K (FSDP, 2 GPUs)
# Customize MODULE, CONFIG, or add --training.steps N via sbatch --wrap

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Create logs dir if missing
mkdir -p logs

export LOGLEVEL=INFO
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR=/scratch/bcjw/jli37/tmp
mkdir -p "$TMPDIR"

# Activate conda env on scratch (avoids installing to quota-limited ~/.local)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate areal


# Optional: uncomment for debugging
# export NCCL_DEBUG=WARN
# export CUDA_LAUNCH_BLOCKING=1

MODULE=${MODULE:-ouro}
CONFIG=${CONFIG:-ouro_1_4b}

./run_train.sh "$@"