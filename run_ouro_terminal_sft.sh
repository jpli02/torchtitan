#!/bin/bash
# Full-backbone SFT of Ouro-1.4B-Thinking on the terminal-agent trajectory
# mixture (Nemotron-Terminal-Corpus + TerminalTraj + OpenThoughts-Agent), 1000
# steps, with W&B monitoring.
#
# Deliberately calls torchtitan.train directly rather than going through
# run_train.sh: that wrapper injects DEFAULT_ARGS (local_batch_size=2,
# seq_len=2048, data_parallel_shard_degree=2) which would silently override the
# batch/sequence shape this config depends on. Agent trajectories need the long
# 8192 context, and a full-backbone 1.4B finetune needs batch size 1 to fit.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3 ./run_ouro_terminal_sft.sh
#   CUDA_VISIBLE_DEVICES=3 ./run_ouro_terminal_sft.sh --training.steps 20   # smoke
set -euo pipefail
cd "$(dirname "$0")"

# HF_TOKEN for dataset access; WANDB_API_KEY comes from ~/.netrc.
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a

export WANDB_PROJECT="${WANDB_PROJECT:-ouro-terminal-sft}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-ouro-1.4b-thinking-terminal-sft-$(date +%m%d_%H%M)}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export LOGLEVEL="${LOGLEVEL:-INFO}"
# HF streaming over a long run hits transient parquet-fetch errors; give the
# hub client room to retry rather than dying mid-training.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

DUMP_FOLDER="${DUMP_FOLDER:-outputs/ouro_terminal_sft}"
CKPT_FOLDER="${CKPT_FOLDER:-/home/jli199/boptim_scratch/ouro_terminal_sft/checkpoint}"
mkdir -p logs "$DUMP_FOLDER"

echo "[sft] wandb: $WANDB_PROJECT / $WANDB_RUN_NAME"
echo "[sft] checkpoints: $CKPT_FOLDER"

# Single GPU: data_parallel_shard_degree=1. The 1.4B backbone plus Adam states
# fits on one 46GB A6000 at batch 1 / seq 8192, and a 1-GPU run sidesteps the
# 2-GPU FSDP SIGABRT this node has hit before on Ouro.
exec "${TORCHRUN:-.venv/bin/torchrun}" \
  --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 \
  -m torchtitan.train \
  --module ouro \
  --config ouro_1_4b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --parallelism.tensor_parallel_degree 1 \
  --dump_folder "$DUMP_FOLDER" \
  --checkpoint.folder "$CKPT_FOLDER" \
  "$@"
