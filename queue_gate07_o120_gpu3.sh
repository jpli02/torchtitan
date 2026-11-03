#!/bin/bash
# When GPU 3 finishes the first-slice gate0.5 point, run the second slice's gate0.7 there and stop the GPU 2
# chain once it has graded gate0.5_o120 (so gate0.7_o120 is not generated twice).
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
REP=$TMP/wallclock_lcb.txt
END=$(( $(date +%s) + 36000 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q '^gate0.5 GRADED' "$REP" 2>/dev/null && break
  sleep 120
done
grep -q '^gate0.5 GRADED' "$REP" 2>/dev/null || { echo "gate0.5 never graded; giving up $(date '+%m-%d %H:%M')"; exit 1; }
cd "$WT" || exit 1
REPORT=$REP MARK='gate0.5_o120 GRADED' SHELL_PAT='bash wallclock_lcb2.sh' ENV_MATCH='OFFSET=120,GPU=2' KILL_PAT='wcl_gate0.7_o120.jsonl' GPU=2 \
  NOTE='GPU 2 chain stopped after gate0.5_o120 (gate0.7_o120 ran on GPU 3)' python3 -I stop_chain_after.py > "$TMP/stop_chain_o120.log" 2>&1 &
GPU=3 OFFSET=120 POINTS="gate0.7" bash wallclock_lcb2.sh
wait
