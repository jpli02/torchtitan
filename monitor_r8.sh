#!/bin/bash
set -u
PY=/home/jli199/torchtitan/.venv/bin/python
CNT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/_r8count.py
PROG=/home/jli199/.claude/jobs/c0d2da0a/tmp/monitor_r8.txt
: > "$PROG"
END=$(( $(date +%s) + 3000 ))
while [ "$(date +%s)" -lt "$END" ]; do
  out=$("$PY" "$CNT")
  r=$(echo "$out" | grep -oE "resolved=[0-9]+" | cut -d= -f2)
  g=$(echo "$out" | grep -oE "graded=[0-9]+" | cut -d= -f2)
  echo "$(date '+%H:%M') $out" >> "$PROG"
  if [ "${r:-0}" -ge 2 ]; then echo "REACHED $out" >> "$PROG"; break; fi
  if [ "${g:-0}" -ge 12 ]; then echo "ALLGRADED $out" >> "$PROG"; break; fi
  sleep 60
done
echo "done $("$PY" "$CNT")" >> "$PROG"
