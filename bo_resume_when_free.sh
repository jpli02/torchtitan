#!/bin/bash
# Resume the pooled LiveCodeBench BO on the first card that is entirely free (>= MINFREE MiB): other users' jobs took cards 2 to 5
# and the epoch evaluations hold card 1. Card 1 becomes a candidate once the epoch chain has written its "finished" line.
# Completed evaluations are replayed from the cache, so a restart loses nothing.
#   bash bo_resume_when_free.sh
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
MINFREE=${MINFREE:-40000}
END=$(( $(date +%s) + 259200 ))
while [ "$(date +%s)" -lt "$END" ]; do
  cands="0 2 3 4 5"
  grep -q 'chain for epochs \[6 7 5 8\] finished' "$TMP/train_tvary_r1_8ep.txt" 2>/dev/null && cands="1 $cands"
  pick=""
  for g in $cands; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [ "${free:-0}" -ge "$MINFREE" ] && [ "${util:-100}" -le 5 ]; then pick=$g; break; fi
  done
  if [ -n "$pick" ]; then
    echo "resuming the pooled BO on gpu $pick $(date '+%m-%d %H:%M')" | tee -a "$TMP/bo_resume.log"
    : > /home/jli199/boptim_scratch/bo_lcb_router_pooled/gpus.txt
    cd /home/jli199/boptim-agent || exit 1
    LCB_BO_GPUS=$pick ./.venv/bin/python -u run_experiment.py --config configs/ouro_lcb_router_pooled_gp.yaml --log-prefix lcb_router_pooled > "$TMP/lcb_bo_pooled_resumed.log" 2>&1
    echo "pooled BO exited $(date '+%m-%d %H:%M')" | tee -a "$TMP/bo_resume.log"
    exit 0
  fi
  sleep 120
done
echo "no free card within 72 h" | tee -a "$TMP/bo_resume.log"
