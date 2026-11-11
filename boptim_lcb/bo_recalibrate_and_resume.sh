#!/bin/bash
# Put every known pooled-BO point on the stratified-calibration speed protocol, then resume the search:
#   1. recalibrate_points.py re-measures the speed-up of the measured and warm-start points on GPU (candidate + full depth back to back)
#      and rewrites evaluations.jsonl;  2. make_warm_start.py rewrites the optimizer's resume state with those speeds;
#   3. run_experiment.py resumes the search on the same card.
#   GPU=0 bash boptim_lcb/bo_recalibrate_and_resume.sh
set -u
GPU=${GPU:-0}
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
cd /home/jli199/boptim-agent || exit 1
echo "recalibration on gpu $GPU start $(date '+%m-%d %H:%M')" | tee -a "$TMP/bo_recal.log"
./.venv/bin/python -u "$WT/boptim_lcb/recalibrate_points.py" --gpu "$GPU" 2>&1 | grep -v Warning | tee -a "$TMP/bo_recal.log"
./.venv/bin/python "$WT/boptim_lcb/make_warm_start.py" 2>&1 | grep -v Warning | tee -a "$TMP/bo_recal.log"
echo "resuming the pooled BO on gpu $GPU $(date '+%m-%d %H:%M')" | tee -a "$TMP/bo_recal.log" "$TMP/bo_resume.log"
: > /home/jli199/boptim_scratch/bo_lcb_router_pooled/gpus.txt
LCB_BO_GPUS=$GPU ./.venv/bin/python -u run_experiment.py --config configs/ouro_lcb_router_pooled_gp.yaml --log-prefix lcb_router_pooled > "$TMP/lcb_bo_pooled_cal32.log" 2>&1
echo "pooled BO exited $(date '+%m-%d %H:%M')" | tee -a "$TMP/bo_recal.log" "$TMP/bo_resume.log"
