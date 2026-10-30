#!/bin/bash
# LiveCodeBench at a 3,072-token budget is mostly a truncation measurement for this model
# (83/120 generations cut off at K=4). Keep K=4 and K=2 for the record, stop after K=2 is
# graded, and cancel the K=1/K=3 follow-ups so GPU 5 goes to the KV-share / guard evals.
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
END=$(( $(date +%s) + 86400 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q '^K=2 GRADED' "$TMP/depth_lcb.txt" 2>/dev/null && break
  sleep 120
done
for p in $(pgrep -f 'bash coding_sweeps.sh'); do kill "$p" 2>/dev/null; done
for p in $(pgrep -f 'bash drop_extrapolation.sh'); do kill "$p" 2>/dev/null; done
for p in $(pgrep -f 'bash depth_sweep.sh'); do
  if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q '^TASK=lcb$'; then kill "$p" 2>/dev/null; fi
done
sleep 3
for p in $(pgrep -f 'depth_eval.py .*--task lcb'); do kill "$p" 2>/dev/null && echo "killed lcb eval $p"; done
echo "lcb stopped after K=2 $(date '+%m-%d %H:%M')" | tee -a "$TMP/depth_lcb.txt"
