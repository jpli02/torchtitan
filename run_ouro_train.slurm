#!/bin/bash
#SBATCH --job-name=ouro_train
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=02:30:00
#SBATCH --output=logs/ouro_train_%j.out
#SBATCH --error=logs/ouro_train_%j.err

# Ouro training on 2 GPUs.
# Override MODULE/CONFIG/NGPU via env at submit time if needed.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

mkdir -p logs

export LOGLEVEL=${LOGLEVEL:-INFO}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TMPDIR=${TMPDIR:-/scratch/bcjw/jli37/tmp}
mkdir -p "$TMPDIR"

# Activate conda env used for training.
source .venv/bin/activate

# Optional debugging knobs:
# export NCCL_DEBUG=WARN
# export CUDA_LAUNCH_BLOCKING=1

export MODULE=${MODULE:-ouro}
export CONFIG=${CONFIG:-ouro_1_4b}
export NGPU=${NGPU:-2}
export PYTHON=${PYTHON:-python3}

./run_train.sh "$@"