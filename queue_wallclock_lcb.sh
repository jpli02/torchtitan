#!/bin/bash
# Start the LiveCodeBench true-exit wall-clock chain on GPU 3 once the gather-mode tail (lcb_tail_gpu3.sh) is done.
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
END=$(( $(date +%s) + 64800 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q 'LCB tail (gpu 3) done' "$TMP/bench_lcb342.txt" 2>/dev/null && break
  sleep 120
done
grep -q 'LCB tail (gpu 3) done' "$TMP/bench_lcb342.txt" 2>/dev/null || { echo "tail never finished; not starting $(date '+%m-%d %H:%M')"; exit 1; }
cd "$WT" || exit 1
GPU=3 N=120 bash wallclock_lcb.sh
