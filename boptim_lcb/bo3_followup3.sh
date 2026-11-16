#!/bin/bash
# Validation of the fill-on search, third part: after the live phase of bo3_followup2.sh has returned, the other three
# routers that hold the floor on the search set (#14, #13, #20) on the live set, so that ALL five are reported there too
# (no selection by held-out score). Waits without touching the GPU; never overlaps the running 4-shard evaluation.
# Start detached:  setsid nohup bash boptim_lcb/bo3_followup3.sh >> $TMP/bo3_followup.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
M=/home/jli199/boptim_scratch/bo_bfcl_router/fill_maps.pt
END=$(( $(date +%s) + 172800 ))
until grep -q '^live validation returned' "$TMP/bo3_followup.log" 2>/dev/null; do
  [ "$(date +%s)" -gt "$END" ] && { echo "third part gave up waiting $(date '+%m-%d %H:%M')"; exit 1; }
  sleep 120
done
# wait until the shards of the previous phase are really gone
while pgrep -u "$(id -un)" -f 'boptim_lcb/lab_eval.py' > /dev/null; do sleep 30; done
cd /home/jli199/boptim-agent || exit 1
echo "live, remaining floor-holding routers start $(date '+%m-%d %H:%M')"
SET=live ROUTERS="0.2386,0.30835,0.46445|maps=$M;0.2386,0.30595,0.47475|maps=$M;0.24005,0.29995,0.4876|maps=$M" \
  ./.venv/bin/python -u "$WT/boptim_lcb/lab_eval.py" >> "$TMP/lab_eval_live.log" 2>&1
echo "live validation, second part, returned $? $(date '+%m-%d %H:%M')"
