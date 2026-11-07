#!/bin/bash
# Command-focused SFT ablation: the take-turns r1 data (112 new rows + 1:1 v3 replay) and the same 896 steps
# (1.67 true epochs) as the r1 run, but the loss on the assistant prose (analysis/plan) is dropped
# (OURO_SFT_PROSE_KEEP=0, see text_datasets.py). Trains, stages the HF checkpoint and evals the 11 test tasks x 2
# through train_cont_4ep.sh; then runs the 22 ep4-solved training tasks at 1,800 s like eval_train22_r1.sh.
#   GPU=1 bash train_cmdonly.sh
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
TAG=${TAG:-ouro_tvary_r1_cmdonly}
export OURO_SFT_PROSE_KEEP=${KEEP:-0.0}
cd "$WT" || exit 1
DATA=$TMP/tvary_r1_new.jsonl TAG=$TAG GPU=${GPU:-1} STEPS=${STEPS:-896} PORT=${PORT:-8171} bash train_cont_4ep.sh
# training-task fit check at the full budget (same 22 tasks as the r1 runs)
POOL=/home/jli199/boptim_scratch/tb_variants/pool
OUT=/srv/scratch/jli199/tb_runs
TASKS=$(cat "$TMP/train_tasks_ep4_resolved.txt")
REP=$TMP/eval_train_tasks.txt
export OURO_SEQ_CAP=12000
echo "=== $TAG on the 22 ep4-solved training tasks x1, 1800 s, GPU ${GPU:-1} $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
GPU=${GPU:-1} GATE=0 MAXTOK=2048 CKPT=/home/jli199/boptim_scratch/${TAG}_clean NAME=train22_${TAG#ouro_tvary_r1_} PORT=${PORT2:-8172} ATT=1 MINFREE=40000 DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=1800 TASKS="$TASKS" \
  bash eval_gated.sh > "$TMP/train22_${TAG#ouro_tvary_r1_}.log" 2>&1
python3 -I tb_progress.py "$OUT" train22_${TAG#ouro_tvary_r1_} | tee -a "$REP"
python3 -I traj_divergence.py train22_${TAG#ouro_tvary_r1_} | tail -n 3 | tee -a "$REP"
echo "train22_${TAG#ouro_tvary_r1_} done $(date '+%m-%d %H:%M')" | tee -a "$REP"
