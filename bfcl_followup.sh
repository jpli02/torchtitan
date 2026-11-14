#!/bin/bash
# Run BFCL points one by one on a card that another user may squeeze (24 GB headroom): a point whose GRADED line is missing after the
# run is retried up to TRIES times, a minute apart.
#   GPU=3 BATCH=2 PTS="direct_q0.5 direct_q0.2 direct_q0.1 direct_steps0.3-0.5-0.5" bash bfcl_followup.sh
#   GPU=4 PTS="true_full true_q0.5 true_q0.3 true_q0.2" bash bfcl_followup.sh
set -u
: "${GPU:?}"; : "${PTS:?}"
TRIES=${TRIES:-3}; NAME=${NAME:-bfcl}
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft || exit 1
for pt in $PTS; do
  n=0
  until grep -q "^GRADED $pt " "$TMP/bench_${NAME}.txt" 2>/dev/null; do
    n=$((n + 1)); [ "$n" -gt "$TRIES" ] && { echo "$pt: gave up after $TRIES tries"; break; }
    GPU=$GPU NAME=$NAME POINTS="$pt" bash bfcl_points.sh > "$TMP/bfcl_followup_${pt}.log" 2>&1
    grep -q "^GRADED $pt " "$TMP/bench_${NAME}.txt" 2>/dev/null || sleep 60
  done
  grep "^GRADED $pt " "$TMP/bench_${NAME}.txt" | tail -n 1 | cut -c1-200
done
echo "bfcl follow-up [$PTS] on gpu $GPU done $(date '+%m-%d %H:%M')"
