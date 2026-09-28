#!/bin/bash
# tb80sft on the full 80 tasks, UNGATED (stock terminus-2), so the number is
# directly comparable to continue10k 4/80, batching5k 4/80, repro11k 3/80,
# sft20k 2/80, pretrained 0/80 -- same harness, same settings, 1 attempt.
#
# Why this run matters: tb80sft scored 7/24 on the 12-task set under the stock
# agent, the best ungated result of the project, and solved processing-pipeline
# which nothing had solved before. The 12-task set has a +-1 noise floor; 80
# tasks is where the on-distribution data claim gets decided.
#
# Clean by construction: the 27 TB-2 tasks sharing a name with our 80 were
# excluded from the training set at build time.
#
# 2 shards on one card (3 shards OOM'd once KV caches grew over long sessions).
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp

gpu=""
for _ in $(seq 1 240); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, '$2+0>=35000 {print $1; exit}')
  [ -n "$gpu" ] && break
  sleep 60
done
if [ -z "$gpu" ]; then
  echo "ABORT - no GPU with >=35GB free after 4h" | tee -a "$TMP/tb80sft80_state.txt"
  exit 1
fi

echo "=== tb80sft 80-task on gpu $gpu, 2 shards ($(date '+%m-%d %H:%M')) ===" \
  | tee -a "$TMP/tb80sft80_state.txt"
MODEL_NAME=tb80sft80 \
CKPT=/home/jli199/boptim_scratch/ouro_tb80sft_clean \
GPU="$gpu" NSHARD=2 BASEPORT=8260 \
  bash "$TMP/eval80_parallel.sh" >> "$TMP/tb80sft80.driver" 2>&1
cat "$TMP/eval80_tb80sft80.txt" 2>/dev/null | tee -a "$TMP/tb80sft80_state.txt"
echo "--- TB80SFT 80-TASK DONE ---" | tee -a "$TMP/tb80sft80_state.txt"
