#!/bin/bash
# Epoch 8 on all 75 training tasks on GPU 0, as soon as the pooled LiveCodeBench BO there has finished.
# Cards 2 to 5 belong to another user's holder (it re-takes them the second its own job ends and leaves 24 GB, too little for a
# 12k-token context); the cards we can rely on are GPU 1 (whole) and GPU 0 (39 GB next to an idle server).
#   bash epoch8_after_bo.sh
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft || exit 1
END=$(( $(date +%s) + 172800 ))
until grep -q 'Completed 1 experiment' "$TMP/lcb_bo_pooled_cal32.log" 2>/dev/null; do
  [ "$(date +%s)" -gt "$END" ] && { echo "BO did not finish within 48 h"; exit 1; }
  sleep 60
done
echo "BO finished; epoch 8 on GPU 0 starts $(date '+%m-%d %H:%M')" | tee -a "$TMP/train_tvary_r1_8ep.txt"
sleep 30
GPU=0 MINFREE=36000 EPOCHS_LIST="8" bash epoch_train75_chain2.sh
