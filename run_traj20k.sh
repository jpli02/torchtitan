#!/bin/bash
# The fair test of the data hypothesis: 20k steps from the PRETRAINED base on
# the single corpus closest to terminal-bench, finished trajectories only.
#
# traj1k (same data, 1k steps) scored 2/24 with partial credit BELOW the no-op
# floor and 5 parse errors -- undertrained on format, not a verdict on the data.
# Every checkpoint that scores 4/80 has 10-20k steps behind it; this matches
# that budget so the comparison is like-for-like.
#
# Data, measured (250 rows/corpus):
#   m-a-p/TerminalTraj          fmt 100%   turns 15.1/13   ends complete 92%
#   Nemotron/skill_based_medium fmt  89%   turns  5.9/6    ends complete 12%
# TerminalTraj is the only corpus in terminus-2 format throughout, with ~2x the
# horizon of the others. Nemotron/medium abandons 88% of its trajectories and
# was 30-45% of every prior mix.
#
# Size: 20,000 rows, ~18.4k after the completion filter. At the 4096-token cap
# 20k steps is ~1.1 epochs -- enough to train, no heavy repetition.
#
# Checkpoints every 5k so the curve can be swept (5k/10k/15k/20k) rather than
# judged from the endpoint alone. STEPS is overridable.
set -u
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a

export CUDA_VISIBLE_DEVICES=${GPU:?GPU must be set explicitly}
export OURO_SFT_MIX=terminaltraj
export OURO_SFT_REQUIRE_COMPLETE=1
unset OURO_INIT_FROM          # pretrained Ouro-1.4B-Thinking
export WANDB_MODE=online
export WANDB_PROJECT=ouro-terminal-sft
export WANDB_RUN_NAME="ouro-terminaltraj-${STEPS:-20000}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=60

DUMP=/home/jli199/boptim_scratch/ouro_traj20k
mkdir -p "$DUMP"
echo "[traj20k] mix=$OURO_SFT_MIX require_complete=$OURO_SFT_REQUIRE_COMPLETE gpu=$CUDA_VISIBLE_DEVICES steps=${STEPS:-20000} $(date '+%m-%d %H:%M')"

exec /home/jli199/torchtitan/.venv/bin/torchrun \
  --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 \
  -m torchtitan.train \
  --module ouro --config ouro_1_4b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --dump_folder "$DUMP" \
  --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps ${STEPS:-20000} \
  --checkpoint.interval 5000 \
  --metrics.log_freq 100 \
  "$@"
