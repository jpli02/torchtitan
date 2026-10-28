#!/bin/bash
# Items 3 + 4 training queue on $GPU, after item 1 (take-turns r1 at 4 epochs) finishes.
#   3: composition arms (student-heavy / teacher-heavy / random, 61 rows each, 4 true epochs)
#   4: 2.6B oracle with REAL turn-2 output (if oracle_real.sh harvested >= 8 rows), 300 steps, no replay
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
GPU=${GPU:-5}
REP=$TMP/comp_queue.txt; : > "$REP"
END=$(( $(date +%s) + 172800 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q "TRAIN_TVARY r1 4ep done" "$TMP/train_tvary_r1_4ep.txt" 2>/dev/null && break
  sleep 300
done
grep -q "TRAIN_TVARY r1 4ep done" "$TMP/train_tvary_r1_4ep.txt" 2>/dev/null || { echo "ABORT item 1 not done in 48h" | tee -a "$REP"; exit 1; }
echo "=== COMP QUEUE start $(date '+%m-%d %H:%M') gpu=$GPU ===" | tee -a "$REP"
cd "$WT" || exit 1
for arm in student teacher random; do
  DATA=$TMP/comp_$arm.jsonl TAG=ouro_comp_$arm GPU=$GPU EPOCHS=4 PORT=8148 bash train_cont_4ep.sh
  echo "arm $arm finished $(date '+%m-%d %H:%M')" | tee -a "$REP"
done
if [ -s "$TMP/oracle_real.jsonl" ] && [ "$(wc -l < "$TMP/oracle_real.jsonl")" -ge 8 ]; then
  DATA=$TMP/oracle_real.jsonl TAG=ouro_oracle_real26 GPU=$GPU REPLAY=0 STEPS=300 PORT=8148 bash train_cont_4ep.sh
  echo "oracle_real26 finished $(date '+%m-%d %H:%M')" | tee -a "$REP"
else
  echo "oracle_real.jsonl missing or < 8 rows; oracle training skipped" | tee -a "$REP"
fi
echo "=== COMP QUEUE done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
for a in student teacher random; do grep -E "^comp_${a}_eval: " "$TMP/train_ouro_comp_$a.txt" 2>/dev/null | tail -1; done | tee -a "$REP"
grep -E "^oracle_real26_eval: " "$TMP/train_ouro_oracle_real26.txt" 2>/dev/null | tail -1 | tee -a "$REP"
echo "compare: r1@1.67ep 9/22 (6 tasks) | r1+r2 7/22 | v3+lever-1 8/22 | 1.4B oracle12 greedy 9/11" | tee -a "$REP"
