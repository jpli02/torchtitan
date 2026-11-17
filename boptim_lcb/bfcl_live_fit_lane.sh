#!/bin/bash
# BFCL, live items: does a cache correction fitted ON LIVE ITEMS help there? (The correction fitted on non-live answers
# costs 19 and 2 items at gate 0.3 / 0.25 on the live set.) A cheap test before any search on a live split:
#   1. fit the maps on 120 live items taken from the items with index % 3 == 0 (the other 900 items are never fitted on);
#   2. real-exit evaluation of gate 0.3 and gate 0.25 with these maps on all 1,350 live items (about 53 minutes each);
#   3. compare on the 900 untouched items with the plain gate and with the non-live maps (bfcl_live_heldout_compare.py).
# GPU 0 next to another user's server: three shards with a memory reserve, as in bo3_followup4.sh.
# Start detached:  setsid nohup bash boptim_lcb/bfcl_live_fit_lane.sh >> $TMP/bfcl_live_fit_lane.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
WORK=/home/jli199/boptim_scratch/bo_bfcl_router
PY=/home/jli199/boptim-agent/.venv/bin/python
M=$WORK/fill_maps_live.pt
cd "$WT" || exit 1
if [ ! -s "$M" ]; then
  echo "live fit starts $(date '+%m-%d %H:%M')"
  CUDA_VISIBLE_DEVICES=0 KV_DATA=/srv/scratch/jli199/bfcl_stage/bfcl_live_all.jsonl KV_RUNS=$WORK/live KV_CAND=third \
    KV_REPORT=$WORK/kv_fill_fit_live.json "$PY" -u boptim_lcb/kv_fill_fit.py 120 40 --save "$M" > "$TMP/kv_fill_fit_live.log" 2>&1
  echo "live fit returned $? $(date '+%m-%d %H:%M')"
fi
[ -s "$M" ] || { echo "LIVE-FIT lane stopped: no maps"; tail -n 5 "$TMP/kv_fill_fit_live.log" | cut -c1-200; exit 1; }
cd /home/jli199/boptim-agent || exit 1
BFCL_SHARD_MIB=11500 BFCL_RESERVE_MIB=4000 BFCL_SHARDS=3 SET=live ROUTERS="0.3,0.3,0.3|maps=$M;0.25,0.25,0.25|maps=$M" \
  ./.venv/bin/python -u "$WT/boptim_lcb/lab_eval.py" >> "$TMP/lab_eval_live.log" 2>&1
echo "live evaluations with the live-fitted maps returned $? $(date '+%m-%d %H:%M')"
cd "$WT" && "$PY" -I boptim_lcb/bfcl_live_heldout_compare.py
echo "LIVE-FIT lane done $(date '+%m-%d %H:%M')"
