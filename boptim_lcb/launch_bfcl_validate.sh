#!/bin/bash
# Held-out validation of the BFCL router search, detached (survives a session restart); restartable, finished points are cached.
#   setsid nohup bash boptim_lcb/launch_bfcl_validate.sh > /dev/null 2>&1 < /dev/null &
# Writes the validation log to $TMP/bfcl_validate.log and a final "validation returned" line to $TMP/bfcl_after_bo.log.
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
STAGE=/srv/scratch/jli199/bfcl_stage
[ -s "$STAGE/bfcl_live_all.jsonl" ] || cp "$TMP/bfcl_live_all.jsonl" "$STAGE/bfcl_live_all.jsonl"
cd /home/jli199/boptim-agent || exit 1
echo "validation (re)started $(date '+%m-%d %H:%M')" >> "$TMP/bfcl_after_bo.log"
BFCL_BO_GPUS=${BFCL_BO_GPUS:-0,5} BFCL_SHARDS=${BFCL_SHARDS:-4} ./.venv/bin/python -u "$WT/boptim_lcb/bfcl_validate.py" >> "$TMP/bfcl_validate.log" 2>&1
echo "validation returned $? $(date '+%m-%d %H:%M')" >> "$TMP/bfcl_after_bo.log"
