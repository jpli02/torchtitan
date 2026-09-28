#!/bin/bash
# continue10k on the full 80 tasks, so it can be compared to sft20k and
# pretrained on the same task set. Until now continue10k had only a 12-task
# number (3/12), which is not comparable to sft20k's 80-task 2/80.
#
# 2 shards, not 3: sft20k at 3 shards threw 1,042 OOM 500s alone on a 45.5GB
# card once KV cache grew over 80 tasks' longer sessions. 2 shards traded ~25%
# throughput for a run whose numbers can be trusted.
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
  echo "ABORT - no GPU with >=27GB free after 2h" | tee -a "$TMP/c10k80_state.txt"
  exit 1
fi

echo "=== continue10k 80-task on gpu $gpu, 2 shards ($(date +%H:%M)) ===" \
  | tee -a "$TMP/c10k80_state.txt"
MODEL_NAME=c10k80 \
CKPT=/home/jli199/boptim_scratch/ouro_c10k_clean \
GPU="$gpu" NSHARD=2 BASEPORT=8220 \
  bash "$TMP/eval80_parallel.sh" >> "$TMP/c10k80.driver" 2>&1
cat "$TMP/eval80_c10k80.txt" 2>/dev/null | tee -a "$TMP/c10k80_state.txt"
echo "--- C10K 80-TASK DONE ---" | tee -a "$TMP/c10k80_state.txt"
