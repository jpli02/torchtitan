#!/bin/bash
# Keep one card busy: poll (10 s) for WAITFOR in the training-task report, then run the given steps in sequence with GPU exported.
# (Another user's holder grabs any card that goes idle for a minute, so the next job must start the moment the previous one ends.)
#   GPU=2 WAITFOR='train75_ep4_1800_s2 done' bash chain_after.sh \
#     'CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_clean TAG=r1 SHARD=2 NSHARD=4 MINFREE=20000 bash eval_train75_sharded2.sh' \
#     'CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_clean TAG=r1 SHARD=4 NSHARD=4 MINFREE=20000 bash eval_train75_sharded2.sh'
set -u
: "${GPU:?}"; : "${WAITFOR:?}"
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
REP=$TMP/eval_train_tasks.txt
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft || exit 1
until grep -q "$WAITFOR" "$REP"; do sleep 10; done
for step in "$@"; do
  echo "chain gpu $GPU start: $step  ($(date '+%m-%d %H:%M'))" | tee -a "$TMP/chains.log"
  GPU=$GPU bash -c "$step"
  echo "chain gpu $GPU step done ($(date '+%m-%d %H:%M'))" | tee -a "$TMP/chains.log"
done
echo "chain gpu $GPU finished $(date '+%m-%d %H:%M')" | tee -a "$TMP/chains.log"
