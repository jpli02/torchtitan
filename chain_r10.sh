#!/bin/bash
# Wait for flash round-10 to yield >=2 resolved (or all 12 graded), harvest +
# append to the deepseek flash dataset (target 300 flash), then stop round-10.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
PY=/home/jli199/torchtitan/.venv/bin/python
PROG=/home/jli199/.claude/jobs/c0d2da0a/tmp/chain_r10.txt
: > "$PROG"
END=$(( $(date +%s) + 3600 ))
while [ "$(date +%s)" -lt "$END" ]; do
  out=$("$PY" "$WT/_r10count.py")
  r=$(echo "$out" | grep -oE "resolved=[0-9]+" | cut -d= -f2)
  g=$(echo "$out" | grep -oE "graded=[0-9]+" | cut -d= -f2)
  echo "$(date '+%H:%M') $out" >> "$PROG"
  { [ "${r:-0}" -ge 2 ] || [ "${g:-0}" -ge 12 ]; } && break
  sleep 45
done
"$PY" "$WT/append_r10.py" >> "$PROG" 2>&1
pkill -9 -f "run-id genflash300_r10" 2>/dev/null
pkill -9 -f "gen_traj.sh" 2>/dev/null
echo "$(date '+%H:%M') done flash=$(wc -l < /home/jli199/boptim_scratch/gen_out/deepseek_flash_generated.jsonl)" >> "$PROG"
