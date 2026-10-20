#!/bin/bash
# Wait for flash round-8 to yield >=2 resolved (or all 12 graded), append the new
# trajectories to the deepseek dataset, stop round-8, then launch the Ouro-2.6B
# verification SFT + Terminal-Bench eval.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
PY=/home/jli199/torchtitan/.venv/bin/python
PROG=/home/jli199/.claude/jobs/c0d2da0a/tmp/chain_verify.txt
: > "$PROG"
END=$(( $(date +%s) + 5400 ))
while [ "$(date +%s)" -lt "$END" ]; do
  out=$("$PY" "$WT/_r8count.py")
  r=$(echo "$out" | grep -oE "resolved=[0-9]+" | cut -d= -f2)
  g=$(echo "$out" | grep -oE "graded=[0-9]+" | cut -d= -f2)
  echo "$(date '+%H:%M') $out" >> "$PROG"
  { [ "${r:-0}" -ge 2 ] || [ "${g:-0}" -ge 12 ]; } && break
  sleep 60
done
echo "$(date '+%H:%M') harvesting round-8" >> "$PROG"
"$PY" "$WT/append_r8.py" >> "$PROG" 2>&1
pkill -9 -f "run-id genflash300_r8" 2>/dev/null
pkill -9 -f "gen_traj.sh" 2>/dev/null
sleep 3
echo "$(date '+%H:%M') launching verification SFT" >> "$PROG"
TAG=ds_verify JSONL=/home/jli199/boptim_scratch/gen_out/deepseek_generated.jsonl STEPS=1000 \
  nohup bash "$WT/verify_ouro_sft.sh" > /home/jli199/.claude/jobs/c0d2da0a/tmp/ds_verify_launch.log 2>&1 &
echo "$(date '+%H:%M') verify SFT pid $!" >> "$PROG"
