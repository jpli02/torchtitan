#!/bin/bash
# One-shot safeguard for the r1+r2 combined TRAINING: wait for it to start, then
# ~25 min later confirm it is stepping using the CORRECT signals (the .wandb run
# file growing + GPU util), not stdout loss lines (which torchtitan does not print
# under these flags -- the earlier false alarm).
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
REP=$TMP/train_tvary_r12.txt
END=$(( $(date +%s) + 345600 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q "training on gpu" "$REP" 2>/dev/null && break
  grep -q "^ABORT" "$REP" 2>/dev/null && { echo "r12 TRAIN ABORTED before start:"; cat "$REP"; exit 0; }
  sleep 300
done
sleep 1500
echo "=== r12 training stepping check $(date '+%m-%d %H:%M') ==="
grep -E "harvested=|mixed=|training on gpu" "$REP" 2>/dev/null
gpu=$(grep -oE "training on gpu [0-9]+" "$REP" | grep -oE "[0-9]+$")
echo "gpu $gpu util/mem: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "${gpu:-0}")"
echo "torchtitan alive: $(pgrep -f 'torchtitan.train' | grep -v pgrep | grep -c .)"
echo "now $(date '+%H:%M:%S'); .wandb run file (size, last write):"
ls -l --time-style=+%H:%M:%S /home/jli199/boptim_scratch/ouro_tvary_r12/tb/*/wandb/offline-run-*/run-*.wandb 2>/dev/null | awk '{print "   "$5" bytes @ "$6}'
echo "(healthy = last write within the last ~2 min and GPU util high; stale file + idle GPU = hang)"
