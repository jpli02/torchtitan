#!/bin/bash
# Single-corpus SFT from the PRETRAINED base: TerminalTraj only, finished
# trajectories only, 1000 steps.
#
# Why this and not another mix. Per-corpus measurement (250 rows each):
#
#   corpus                          fmt%   turns mean/med   ends complete
#   Nemotron/skill_based_medium      89%       5.9 / 6           12%
#   Nemotron/skill_based_easy        87%       7.4 / 7           97%
#   m-a-p/TerminalTraj              100%      15.1 / 13          92%
#   OpenThoughts-Agent-v1-SFT        99%       6.0 / 6          100%
#
# skill_based_medium abandons 88% of its trajectories and carried 30-45% weight
# in every previous mix, so roughly a third of all training signal to date was
# sessions that give up. TerminalTraj is the only corpus at 100% terminus-2
# format -- the shape terminal-bench's agent actually parses -- with ~2x the
# horizon of the others.
#
# Combined with OURO_SFT_REQUIRE_COMPLETE=1 the stream keeps 92% of rows at a
# median of 15 assistant turns, against 8 for the old longhorizon mix and 22 for
# the benchmark itself.
#
# From the BASE, not continue10k: every checkpoint so far descends from mixes
# dominated by the abandoning corpus, so continuing from one would carry that
# behaviour in. 1000 steps first as a cheap read before spending more.
set -u
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a

export CUDA_VISIBLE_DEVICES=${GPU:?GPU must be set explicitly}
export OURO_SFT_MIX=terminaltraj
export OURO_SFT_REQUIRE_COMPLETE=1
unset OURO_INIT_FROM          # start from the pretrained Ouro-1.4B-Thinking
export WANDB_MODE=online
export WANDB_PROJECT=ouro-terminal-sft
export WANDB_RUN_NAME="ouro-terminaltraj-1k"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=60

DUMP=/home/jli199/boptim_scratch/ouro_traj_sft
mkdir -p "$DUMP"
echo "[traj] mix=$OURO_SFT_MIX require_complete=$OURO_SFT_REQUIRE_COMPLETE gpu=$CUDA_VISIBLE_DEVICES steps=${STEPS:-1000}"

exec /home/jli199/torchtitan/.venv/bin/torchrun \
  --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 \
  -m torchtitan.train \
  --module ouro --config ouro_1_4b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --dump_folder "$DUMP" \
  --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps ${STEPS:-1000} \
  --checkpoint.interval 500 \
  --metrics.log_freq 50 \
  "$@"
