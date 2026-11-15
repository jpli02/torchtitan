#!/bin/bash
# Hand-off on the one BFCL card: held-out part of the validation -> second search (exit shares) -> rest of the validation.
#   1. wait until the running validation has finished the held-out set (575 unseen items);
#   2. pause it before the long live set (1,350 items; restartable, finished points are cached);
#   3. run the second search, bfcl_bo_run2.py (20 evaluations on the search set);
#   4. start the validation again: it adds the second search's best distinct routers and runs the live set once, for all of them.
# Start detached:  setsid nohup bash boptim_lcb/bfcl_round2_chain.sh > $TMP/bfcl_round2_chain.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
VLOG=$TMP/bfcl_validate.log
until grep -q 'set heldout done' "$VLOG" 2>/dev/null; do
  pgrep -u "$USER" -f bfcl_validate.py > /dev/null || { echo "the validation is gone before the held-out set finished $(date '+%m-%d %H:%M')"; break; }
  sleep 30
done
echo "held-out set done; validation paused for the second search $(date '+%m-%d %H:%M')"
python3 -I "$WT/kill_by_pattern.py" launch_bfcl_validate.sh bfcl_validate.py bfcl_stage/bfcl_live_all.jsonl
cd /home/jli199/boptim-agent || exit 1
cp "$WT/boptim_lcb/ouro_bfcl_router.py" objective/ouro_bfcl_router.py
echo "=== second search started $(date '+%m-%d %H:%M') ===" >> "$TMP/bfcl_bo2.log"
BFCL_BO_GPUS=${BFCL_BO_GPUS:-0,5} BFCL_SHARDS=${BFCL_SHARDS:-4} BO_ITERS=${BO_ITERS:-10} ./.venv/bin/python -u "$WT/boptim_lcb/bfcl_bo_run2.py" >> "$TMP/bfcl_bo2.log" 2>&1
echo "second search returned $? $(date '+%m-%d %H:%M'); validation resumes"
TOP=2 bash "$WT/boptim_lcb/launch_bfcl_validate.sh"
echo "chain finished $(date '+%m-%d %H:%M')"
