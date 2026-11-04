#!/bin/bash
# Sampling-vs-fit test: the same 22 ep4-solved training tasks as eval_train22_r1.sh, but GREEDY decoding (TEMP=0)
# and a 2,048-token cap per turn (MAXTOK, prevents the runaway-generation cascade seen on 11-03). If r1 replays its
# demonstrated trajectories more often and solves more at temperature 0, the training-task gap is sampling, not fit.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
POOL=/home/jli199/boptim_scratch/tb_variants/pool
OUT=/srv/scratch/jli199/tb_runs
TASKS=$(cat "$TMP/train_tasks_ep4_resolved.txt")
REP=$TMP/eval_train_tasks.txt
cd "$WT" || exit 1
export OURO_SEQ_CAP=12000
echo "=== r1 GREEDY (temp 0, 2,048-token cap) on the 22 ep4-solved training tasks x1, 1800 s, GPU ${GPU:-5} $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
GPU=${GPU:-5} GATE=0 TEMP=0 MAXTOK=2048 CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_clean NAME=train22_r1_greedy PORT=8166 ATT=1 MINFREE=40000 DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=1800 TASKS="$TASKS" \
  bash eval_gated.sh > "$TMP/train22_r1_greedy.log" 2>&1
python3 -I tb_progress.py "$OUT" train22_r1_greedy | tee -a "$REP"
python3 -I traj_divergence.py train22_r1_greedy | tail -n 3 | tee -a "$REP"
echo "train22_r1_greedy done $(date '+%m-%d %H:%M')" | tee -a "$REP"
