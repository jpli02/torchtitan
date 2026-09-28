#!/bin/bash
# repro11k on the full 80 tasks.
#
# repro11k and continue10k tied on the 12-task set (0.208, 3/12) but solved
# DIFFERENT third tasks -- fibonacci-server vs heterogeneous-dates. Running both
# at 80 tasks tests whether that agreement holds at scale, which doubles as the
# seed-variance measurement this project has never had: two runs of the same
# recipe, same eval, so the spread between them is the noise floor every other
# comparison should be judged against.
#
# 2 shards for the same reason as the other 80-task runs: sft20k at 3 shards
# threw 1,042 OOM 500s alone on a 45.5GB card once KV cache grew.
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
  echo "ABORT - no GPU with >=27GB free after 2h" | tee -a "$TMP/repro80_state.txt"
  exit 1
fi

echo "=== repro11k 80-task on gpu $gpu, 2 shards ($(date +%H:%M)) ===" \
  | tee -a "$TMP/repro80_state.txt"
MODEL_NAME=repro80 \
CKPT=/home/jli199/boptim_scratch/ouro_repro_clean \
GPU="$gpu" NSHARD=2 BASEPORT=8230 \
  bash "$TMP/eval80_parallel.sh" >> "$TMP/repro80.driver" 2>&1
cat "$TMP/eval80_repro80.txt" 2>/dev/null | tee -a "$TMP/repro80_state.txt"
echo "--- REPRO 80-TASK DONE ---" | tee -a "$TMP/repro80_state.txt"
