#!/bin/bash
# batching5k on the full 80 tasks -- the missing number. It has only ever been
# scored on the 12-task subset (4/24 pass@2), which is not comparable to
# continue10k's 4/80, repro11k's 3/80, sft20k's 2/80 or pretrained's 0/80.
#
# UNGATED on purpose: this is the baseline the task_complete verification gate
# will be measured against, so it must run the stock terminus-2 agent.
#
# 2 shards, not 3: sft20k at 3 shards threw 1,042 OOM 500s on a 45.5GB card once
# KV cache grew over 80 tasks' longer sessions.
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp

gpu=""
for _ in $(seq 1 120); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, '$2+0>=27000 {print $1; exit}')
  [ -n "$gpu" ] && break
  sleep 60
done
if [ -z "$gpu" ]; then
  echo "ABORT - no GPU with >=27GB free after 2h" | tee -a "$TMP/batching80_state.txt"
  exit 1
fi

echo "=== batching5k 80-task on gpu $gpu, 2 shards ($(date '+%m-%d %H:%M')) ===" \
  | tee -a "$TMP/batching80_state.txt"
MODEL_NAME=batching80 \
CKPT=/home/jli199/boptim_scratch/ouro_batching_clean \
GPU="$gpu" NSHARD=2 BASEPORT=8250 \
  bash "$TMP/eval80_parallel.sh" >> "$TMP/batching80.driver" 2>&1
cat "$TMP/eval80_batching80.txt" 2>/dev/null | tee -a "$TMP/batching80_state.txt"
echo "--- BATCHING5K 80-TASK DONE ---" | tee -a "$TMP/batching80_state.txt"
