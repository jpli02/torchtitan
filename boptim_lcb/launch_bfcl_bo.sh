#!/bin/bash
# Detached launcher for the BFCL router BO, so the run survives a restart of the session that started it.
#   bash boptim_lcb/launch_bfcl_bo.sh            (BFCL_BO_GPUS=0,5 BFCL_SHARDS=4 BO_ITERS=10 by default)
# Finished evaluations are reused from the work directory, so starting it again continues the run.
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
cd /home/jli199/boptim-agent || exit 1
cp "$WT/boptim_lcb/ouro_bfcl_router.py" objective/ouro_bfcl_router.py
export BFCL_BO_GPUS=${BFCL_BO_GPUS:-0,5} BFCL_SHARDS=${BFCL_SHARDS:-4} BO_ITERS=${BO_ITERS:-10}
LOG=${LOG:-/home/jli199/.claude/jobs/c0d2da0a/tmp/bfcl_bo.log}
echo "=== driver (re)started $(date '+%m-%d %H:%M') gpus=$BFCL_BO_GPUS shards=$BFCL_SHARDS iters=$BO_ITERS ===" >> "$LOG"
setsid nohup ./.venv/bin/python -u "$WT/boptim_lcb/bfcl_bo_run.py" >> "$LOG" 2>&1 < /dev/null &
echo "driver pid $!"
