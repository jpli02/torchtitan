#!/bin/bash
# Wait for flash round-9 to yield >=10 resolved (or all 50 graded), harvest +
# append the new trajectories to the deepseek flash dataset (target 300 flash),
# then stop round-9.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
PY=/home/jli199/torchtitan/.venv/bin/python
PROG=/home/jli199/.claude/jobs/c0d2da0a/tmp/chain_r9.txt
: > "$PROG"
END=$(( $(date +%s) + 7200 ))
while [ "$(date +%s)" -lt "$END" ]; do
  out=$("$PY" "$WT/_r9count.py")
  r=$(echo "$out" | grep -oE "resolved=[0-9]+" | cut -d= -f2)
  g=$(echo "$out" | grep -oE "graded=[0-9]+" | cut -d= -f2)
  echo "$(date '+%H:%M') $out" >> "$PROG"
  { [ "${r:-0}" -ge 10 ] || [ "${g:-0}" -ge 50 ]; } && break
  sleep 60
done
echo "$(date '+%H:%M') harvesting round-9" >> "$PROG"
"$PY" "$WT/append_r9.py" >> "$PROG" 2>&1
pkill -9 -f "run-id genflash300_r9" 2>/dev/null
pkill -9 -f "gen_traj.sh" 2>/dev/null
echo "$(date '+%H:%M') done flash=$(wc -l < /home/jli199/boptim_scratch/gen_out/deepseek_flash_generated.jsonl)" >> "$PROG"
