#!/bin/bash
# Re-grade existing BFCL generations with the current grader (no GPU): rewrites the GRADED lines of $TMP/bench_${NAME}.txt.
#   NAME=bfcl DATA=bfcl400 bash bfcl_regrade.sh
set -u
NAME=${NAME:-bfcl}; DATA=${DATA:-bfcl400}
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
STAGE=/srv/scratch/jli199/bfcl_stage
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
REP=$TMP/bench_${NAME}.txt
cp "$WT/bfcl_tool.py" "$STAGE/"
cp "$REP" "$REP.pregrade.$(date +%H%M)" 2>/dev/null
grep -v '^GRADED ' "$REP" > "$REP.tmp" 2>/dev/null
for f in "$TMP"/bench_${NAME}_*.jsonl; do
  label=$(basename "$f" .jsonl); label=${label#bench_${NAME}_}
  cp "$f" "$STAGE/"
  docker run --rm --network none --memory 8g --cpus 4 -v "$STAGE":/w bfcl-grader python /w/bfcl_tool.py grade "/w/$DATA.jsonl" \
    "/w/bench_${NAME}_${label}.jsonl" "/w/bench_${NAME}_${label}.graded.json" 2>&1 | grep '^GRADED' | sed "s/^GRADED /GRADED $label /" | tee -a "$REP.tmp"
  cp "$STAGE/bench_${NAME}_${label}.graded.json" "$TMP/" 2>/dev/null
done
mv "$REP.tmp" "$REP"
