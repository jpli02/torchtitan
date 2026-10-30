#!/bin/bash
# After the coding_sweeps LCB chain (K=4,2,1,6) finishes, run LiveCodeBench at K=3 on GPU 5
# and grade it (the 2.6B-Thinking model often peaks at T=3 in the Ouro paper, Table 13).
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
END=$(( $(date +%s) + 172800 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q 'lcb sweep done' "$TMP/coding_gpu5.log" 2>/dev/null && break
  sleep 300
done
grep -q 'lcb sweep done' "$TMP/coding_gpu5.log" 2>/dev/null || { echo "ABORT lcb chain not done in 48h"; exit 1; }
S=/tmp/lcb_grade_stage; mkdir -p "$S"
for K in 3; do
  TASK=lcb DATA=$TMP/lcb_em120.jsonl N=120 STEPS="$K" BATCH=4 MAXNEW=3072 GPU=5 bash depth_sweep.sh
  cp "$WT/lcb_grade.py" "$TMP/lcb_em120.jsonl" "$TMP/depth_lcb_k${K}.jsonl" "$S"/
  docker run --rm --network none --memory 4g --cpus 4 -v "$S":/w python:3.12-slim \
    python /w/lcb_grade.py /w/lcb_em120.jsonl "/w/depth_lcb_k${K}.jsonl" "/w/depth_lcb_k${K}.graded.json" 2>&1 | grep -E '^GRADED' | sed "s/^/K=$K /" | tee -a "$TMP/depth_lcb.txt"
  cp "$S/depth_lcb_k${K}.graded.json" "$TMP/" 2>/dev/null
done
echo "lcb K=3 done $(date '+%m-%d %H:%M')"
