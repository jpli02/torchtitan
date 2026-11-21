#!/bin/bash
# The optimizer-comparison queue of the night of 2026-11-20 on the two 9 GB card shares (cards 8 and 9), in priority order.
# Seeds are disjoint between the cards (a run directory is recreated when its run starts, so two cards must never share one).
#   bash randrouter/night_chain.sh 9      router NAS (the paper's scenario): four arms, seeds 1-4
#   bash randrouter/night_chain.sh 8      waits for the exact-cache evaluation that still holds card 8, then the threshold
#                                         space (random search and Gaussian process, 5 seeds; LLM arms, 3 repeats), then NAS seed 5
set -u
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft || exit 1
CARD=$1
NAS_ARMS="random gp_hedge luna-informed luna-blind"
if [ "$CARD" = 9 ]; then
  GPU=9 TAG=nas6 ARMS="$NAS_ARMS" SEEDS="1 2 3 4" ITERS=40 bash randrouter/nas_compare.sh
elif [ "$CARD" = 8 ]; then
  WAIT_PID=${WAIT_PID:-} GPU=8 SPACE=thr OPTIMS="random gp_hedge" SEEDS="1 2 3 4 5" ITERS=30 bash randrouter/opt_compare.sh
  GPU=8 SPACE=thr ARMS="informed blind" REPS="1 2 3" ITERS=30 bash randrouter/opt_compare_llm.sh
  GPU=8 TAG=nas6 ARMS="$NAS_ARMS" SEEDS="5" ITERS=40 bash randrouter/nas_compare.sh
fi
echo "NIGHT CHAIN card $CARD done $(date '+%m-%d %H:%M')"
