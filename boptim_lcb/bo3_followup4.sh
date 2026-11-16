#!/bin/bash
# Validation of the fill-on search on GPU 0, restarted 11-15 23:50 with a smaller memory footprint (replaces the rest of
# bo3_followup2.sh and all of bo3_followup3.sh). Another user runs jobs on this card: the shard count now leaves
# BFCL_RESERVE_MIB free and budgets the live set's real peak per shard (11.5 GB), so two shards run on the live set while
# their jobs are present and three when the card is otherwise idle.
#   A. live set (1,350): #12 + correction, plain gate 0.25, #14 + correction, gate 0.3 + correction
#   B. held-out ABLATION: the searched routers WITHOUT the cache correction (#12, #14, #17, #20) and the plain gate 0.2443
#   C. live set: plain gate 0.2, gate 0.2 + correction, #17, #20, #13 (all + correction)
# Results already in the cache (gate 0.25 + correction on live) are skipped by the objective.
# Start detached:  setsid nohup bash boptim_lcb/bo3_followup4.sh >> $TMP/bo3_followup.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
M=/home/jli199/boptim_scratch/bo_bfcl_router/fill_maps.pt
R12="0.2372,0.29355,0.4755"
R13="0.2386,0.30595,0.47475"
R14="0.2386,0.30835,0.46445"
R17="0.2358,0.20185,0.5019"
R20="0.24005,0.29995,0.4876"
export BFCL_RESERVE_MIB=4000
cd /home/jli199/boptim-agent || exit 1
echo "live part A start $(date '+%m-%d %H:%M')"
BFCL_SHARD_MIB=11500 BFCL_SHARDS=3 SET=live ROUTERS="$R12|maps=$M;0.25,0.25,0.25;$R14|maps=$M;0.3,0.3,0.3|maps=$M" \
  ./.venv/bin/python -u "$WT/boptim_lcb/lab_eval.py" >> "$TMP/lab_eval_live.log" 2>&1
echo "live validation returned $? $(date '+%m-%d %H:%M')"
echo "held-out ablation (searched routers without the correction) start $(date '+%m-%d %H:%M')"
BFCL_SHARD_MIB=9500 BFCL_SHARDS=4 SET=heldout ROUTERS="$R12;$R14;0.2443,0.2443,0.2443;$R17;$R20" \
  ./.venv/bin/python -u "$WT/boptim_lcb/lab_eval.py" >> "$TMP/lab_eval_heldout.log" 2>&1
echo "held-out ablation returned $? $(date '+%m-%d %H:%M')"
echo "live part C start $(date '+%m-%d %H:%M')"
BFCL_SHARD_MIB=11500 BFCL_SHARDS=3 SET=live ROUTERS="0.2,0.2,0.2;0.2,0.2,0.2|maps=$M;$R17|maps=$M;$R20|maps=$M;$R13|maps=$M" \
  ./.venv/bin/python -u "$WT/boptim_lcb/lab_eval.py" >> "$TMP/lab_eval_live.log" 2>&1
echo "live validation, second part, returned $? $(date '+%m-%d %H:%M')"
