#!/bin/bash
# Runs the held-out validation as soon as the BFCL router search ends (detached; survives a session restart):
#   setsid nohup bash boptim_lcb/bfcl_after_bo.sh > $TMP/bfcl_after_bo.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
STAGE=/srv/scratch/jli199/bfcl_stage
LOG=$TMP/bfcl_bo.log
until grep -q 'BO finished with code' "$LOG" 2>/dev/null; do
  pgrep -u "$USER" -f bfcl_bo_run.py > /dev/null || { echo "the BO driver is gone without a finish line $(date '+%m-%d %H:%M')"; break; }
  sleep 60
done
echo "search ended ($(grep 'BO finished with code' "$LOG" | tail -n 1)); validation starts $(date '+%m-%d %H:%M')"
[ -s "$STAGE/bfcl_live_all.jsonl" ] || cp "$TMP/bfcl_live_all.jsonl" "$STAGE/bfcl_live_all.jsonl"
cd /home/jli199/boptim-agent || exit 1
BFCL_BO_GPUS=${BFCL_BO_GPUS:-0,5} BFCL_SHARDS=${BFCL_SHARDS:-4} ./.venv/bin/python -u "$WT/boptim_lcb/bfcl_validate.py" >> "$TMP/bfcl_validate.log" 2>&1
echo "validation returned $? $(date '+%m-%d %H:%M')"
