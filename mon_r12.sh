#!/bin/bash
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
END=$(( $(date +%s) + 345600 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -qE "TRAIN_TVARY r12 done|^ABORT" "$TMP/train_tvary_r12.txt" 2>/dev/null && break
  sleep 300
done
echo "=== TAKE-TURNS r1+r2 COMBINED: TRAIN + EVAL ==="
cat "$TMP/train_tvary_r12.txt" 2>/dev/null
echo "--- wandb run: $(grep -rhoE 'offline-run-[0-9_]+-[a-z0-9]+' /home/jli199/terminal_bench_eval/logs/ouro_tvary_r12_train.log 2>/dev/null | head -1) ---"
echo "compare: r1 take-turns 9/22 (6 tasks) | v3+lever-1 8/22 | gated 9/22 | baseline 6/22"
