#!/bin/bash
# Complete CRUXEval to all 800 items for the three headline configurations (items 500-799),
# on GPU 5 after the clean ep4/ep2 Terminal-Bench evals finish there.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
END=$(( $(date +%s) + 172800 ))
while [ "$(date +%s)" -lt "$END" ]; do
  if grep -q 'EP4/EP2 clean eval done' "$TMP/eval_ep4_clean.txt" 2>/dev/null; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 5 | tr -d ' ')
    [ "${free:-0}" -ge 40000 ] && break
  fi
  sleep 300
done
echo "CRUX items 500-799 starting on GPU 5 $(date '+%m-%d %H:%M')"
TASK=crux DATA=$TMP/cruxeval.jsonl N=300 OFFSET=500 THRS="none 0.3 0.1,0.5,0.5" GPU=5 bash gate_sweep.sh
