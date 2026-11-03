#!/bin/bash
# Start GSM-Symbolic true-exit points on a card once a LiveCodeBench wall-clock line appears (the card frees up):
#   MARK='gate0.7_o240 GRADED' GPU=5 POINTS="gate0.5" bash queue_gsm_after.sh
set -u
: "${MARK:?}"; : "${GPU:?}"; : "${POINTS:?}"
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
REPORT=${REPORT:-$TMP/wallclock_lcb.txt}   # the report file to watch for MARK
END=$(( $(date +%s) + 36000 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q "^$MARK" "$REPORT" 2>/dev/null && break
  sleep 120
done
grep -q "^$MARK" "$REPORT" 2>/dev/null || { echo "$MARK never appeared; not starting $(date '+%m-%d %H:%M')"; exit 1; }
sleep 30
cd "$WT" || exit 1
GPU=$GPU POINTS="$POINTS" bash wallclock_gsm.sh
