#!/bin/bash
# +5k steps from continue10k on a BATCHING-DENSE mix.
#
# Diagnosis this addresses: the served model emits exactly 1 command per turn
# (1682/1696 turns, never more). Probing its own next-token distribution at the
# batching decision gives P(start another command) = 0.00027 against ~0.195
# implied by the data -- ~700x under. Token-level CE learned the singleton mode
# and SFT sharpened it until the tail vanished. That caps the agent at ~21
# shell commands per task inside the 420s budget, and 65/80 tasks fail on
# agent_timeout at ~21 turns with 0% repeated responses.
#
# OURO_SFT_MIN_CMDS=1.5 keeps only trajectories averaging >=1.5 commands per
# assistant turn: 36% of rows, shifting the target from median 0.96 -> 4.00
# commands/turn. Filtering whole ROWS (not individual turns) is deliberate --
# batching is a trajectory-level habit, and the model conditions on its own
# history, which is what made the collapse self-reinforcing.
#
# LR 1e-5: continuation onto an already-converged checkpoint (grad-norm ~2.5),
# not a fresh run.
#
# Success is checked by re-probing P(batch) -- a 2-minute measurement -- before
# spending an 80-task eval on this.
set -u
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a

export CUDA_VISIBLE_DEVICES=${GPU:?GPU must be set explicitly}
export OURO_SFT_MIX=longhorizon
export OURO_SFT_MIN_CMDS=1.5
export OURO_INIT_FROM=/home/jli199/boptim_scratch/ouro_c10k_clean
export WANDB_MODE=online
export WANDB_PROJECT=ouro-terminal-sft
export WANDB_RUN_NAME="ouro-batching-dense-5k"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=60

DUMP=/home/jli199/boptim_scratch/ouro_batching_sft
mkdir -p "$DUMP"
echo "[batching] mix=$OURO_SFT_MIX min_cmds=$OURO_SFT_MIN_CMDS init=$OURO_INIT_FROM gpu=$CUDA_VISIBLE_DEVICES"

exec /home/jli199/torchtitan/.venv/bin/torchrun \
  --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 \
  -m torchtitan.train \
  --module ouro --config ouro_1_4b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --dump_folder "$DUMP" \
  --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps ${STEPS:-5000} \
  --optimizer.lr 1e-5 \
  --checkpoint.interval 1000 \
  --metrics.log_freq 50 \
  "$@"
