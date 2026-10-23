#!/bin/bash
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
END=$(( $(date +%s) + 259200 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -qE "TEACHERVARY r2 done|^ABORT" "$TMP/teachervary_round2.txt" 2>/dev/null && break
  sleep 300
done
echo "=== TAKE-TURNS ROUND 2 (clean, new key) GENERATION DONE ==="
cat "$TMP/teachervary_round2.txt" 2>/dev/null
tt=0; tc=0
for k in 0 1 2 3 4 5; do
  L=/home/jli199/terminal_bench_eval/logs/tvary_r2_sh${k}_server.log
  tt=$((tt+$(grep -c 'TEACHER turn' "$L" 2>/dev/null))); tc=$((tc+$(grep -c 'teacher call failed' "$L" 2>/dev/null)))
done
echo "--- teacher turns fired: $tt   failures: $tc (must be ~0) ---"
