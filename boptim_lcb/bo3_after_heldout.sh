#!/bin/bash
# Starts the fill-on router search (bfcl_bo_run3.py) when the held-out check of the calibrated fill is in, and only if it holds
# there: gate 0.2 with the fill must beat the plain gate 0.2 on the 575 unseen items (74.96 %) by at least one point.
#   setsid nohup bash boptim_lcb/bo3_after_heldout.sh > $TMP/bo3_after_heldout.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
HL=$TMP/lab_eval_heldout.log
until grep -q 'lab eval finished\|Traceback' "$HL" 2>/dev/null; do sleep 30; done
acc=$(grep '^LAB ' "$HL" | grep '"router": "0.2,0.2,0.2|maps' | grep -o '"acc_pct": [0-9.]*' | grep -o '[0-9.]*$' | tail -n 1)
echo "held-out gate 0.2 with the fill: ${acc:-missing} % (plain 74.96 %) $(date '+%m-%d %H:%M')"
if [ -n "$acc" ] && python3 -c "import sys; sys.exit(0 if float('$acc') >= 75.96 else 1)"; then
  cd /home/jli199/boptim-agent || exit 1
  echo "=== fill-on search started $(date '+%m-%d %H:%M') ===" >> "$TMP/bfcl_bo3.log"
  BFCL_BO_GPUS=${BFCL_BO_GPUS:-0,5} BFCL_SHARDS=${BFCL_SHARDS:-4} BO_ITERS=${BO_ITERS:-10} ./.venv/bin/python -u "$WT/boptim_lcb/bfcl_bo_run3.py" >> "$TMP/bfcl_bo3.log" 2>&1
  echo "fill-on search returned $? $(date '+%m-%d %H:%M')"
else
  echo "the fill does not hold on the held-out set; search not started"
fi
