#!/bin/bash
# After the fill-on search (bfcl_bo_run3.py) ends, validate on unseen items, one evaluation at a time on the BFCL card:
#   1. held-out set (575): the search's cheapest distinct routers that hold the search floor (corrected cache entries);
#      the released-gate patterns with and without the correction are already there;
#   2. live set (1,350): the released-gate patterns 0.3 / 0.25 / 0.2 with the correction, the search's best router, and the
#      plain gate at 0.25 and 0.2 (plain full depth and 0.3 are already there).
# Start detached:  setsid nohup bash boptim_lcb/bo3_followup.sh > $TMP/bo3_followup.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
M=/home/jli199/boptim_scratch/bo_bfcl_router/fill_maps.pt
until grep -q 'BO3 finished with code' "$TMP/bfcl_bo3.log" 2>/dev/null; do
  pgrep -u "$USER" -f bfcl_bo_run3.py > /dev/null || { echo "the search driver is gone without a finish line $(date '+%m-%d %H:%M')"; break; }
  sleep 60
done
cd /home/jli199/boptim-agent || exit 1
BEST=$(./.venv/bin/python "$WT/boptim_lcb/bo3_pick.py" 3 2>/dev/null | tail -n 1)
echo "search ended $(date '+%m-%d %H:%M'); routers to validate: ${BEST//$M/MAPS}"
if [ -n "$BEST" ]; then
  SET=heldout ROUTERS="$BEST" ./.venv/bin/python -u "$WT/boptim_lcb/lab_eval.py" >> "$TMP/lab_eval_heldout.log" 2>&1
  echo "held-out validation returned $? $(date '+%m-%d %H:%M')"
fi
FIRST=${BEST%%;*}
SET=live ROUTERS="0.25,0.25,0.25|maps=$M;${FIRST:+$FIRST;}0.3,0.3,0.3|maps=$M;0.2,0.2,0.2|maps=$M;0.25,0.25,0.25;0.2,0.2,0.2" \
  ./.venv/bin/python -u "$WT/boptim_lcb/lab_eval.py" >> "$TMP/lab_eval_live.log" 2>&1
echo "live validation returned $? $(date '+%m-%d %H:%M')"
