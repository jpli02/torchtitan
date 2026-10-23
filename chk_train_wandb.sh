#!/bin/bash
# One-shot: wait for round-1 training to START, then ~8 min later confirm it is
# alive and wandb is actually logging (the user asked for wandb on these runs).
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
END=$(( $(date +%s) + 86400 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q "training on gpu" "$TMP/train_tvary_r1.txt" 2>/dev/null && break
  grep -q "^ABORT" "$TMP/train_tvary_r1.txt" 2>/dev/null && { echo "TRAIN ABORTED before start:"; cat "$TMP/train_tvary_r1.txt"; exit 0; }
  sleep 120
done
sleep 480
echo "=== round-1 training health $(date '+%m-%d %H:%M') ==="
grep -E "harvested=|mixed=|training on gpu" "$TMP/train_tvary_r1.txt" 2>/dev/null
L=/home/jli199/terminal_bench_eval/logs/ouro_tvary_r1_train.log
echo "torchtitan alive: $(pgrep -f 'torchtitan.train' | grep -v pgrep | grep -c .)"
echo "wandb init line: $(grep -c 'Tracking run with wandb' "$L" 2>/dev/null)   wandb run dir: $(grep -oE 'offline-run-[0-9_]+-[a-z0-9]+' "$L" 2>/dev/null | head -1)"
echo "loss lines so far: $(grep -c 'loss:' "$L" 2>/dev/null)   latest: $(grep -oE 'step: *[0-9]+ *loss: *[0-9.]+' "$L" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -1)"
echo "errors: $(grep -ciE 'Traceback|Error' "$L" 2>/dev/null)"
