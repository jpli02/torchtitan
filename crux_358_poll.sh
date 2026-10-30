#!/bin/bash
# CRUXEval depths 3, 5, 8 (n=200): wait for a GPU with >= 40 GB free other than GPU 5
# (the LCB chain), then run the sweep there. GPU 3 was taken by another user mid-queue.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
gpu=""
END=$(( $(date +%s) + 172800 ))
while [ "$(date +%s)" -lt "$END" ]; do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' \
        | awk -F, '$1!=5 && $2+0>=40000 {print $2+0, $1}' | sort -rn | head -1 | awk '{print $2}')
  [ -n "$gpu" ] && break
  sleep 120
done
[ -n "$gpu" ] || { echo "ABORT: no free GPU in 48h"; exit 1; }
echo "crux 3/5/8 starting on gpu $gpu $(date '+%m-%d %H:%M')"
TASK=crux DATA=$TMP/cruxeval.jsonl N=200 STEPS="3 5 8" BATCH=6 MAXNEW=2048 GPU=$gpu bash depth_sweep.sh
