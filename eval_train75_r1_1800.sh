#!/bin/bash
# Pass rate of the take-turns r1 checkpoint on ITS OWN training set: all 75 tasks of train_tasks_r1.txt at the standard
# 1,800 s budget (temperature 0.7, 2,048-token turn cap, 18 h harness cap), split into two shards so two cards finish in
# ~10 h.   SHARD=a GPU=4 bash eval_train75_r1_1800.sh   /   SHARD=b GPU=5 bash eval_train75_r1_1800.sh
set -u
: "${SHARD:?}"; : "${GPU:?}"
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
POOL=/home/jli199/boptim_scratch/tb_variants/pool
OUT=/srv/scratch/jli199/tb_runs
REP=$TMP/eval_train_tasks.txt
cd "$WT" || exit 1
ALL=$(cat "$TMP/train_tasks_r1.txt")
if [ "$SHARD" = a ]; then TASKS=$(echo $ALL | tr ' ' '\n' | sed -n '1,38p' | tr '\n' ' '); PORT=8181; else TASKS=$(echo $ALL | tr ' ' '\n' | sed -n '39,75p' | tr '\n' ' '); PORT=8182; fi
export OURO_SEQ_CAP=12000
echo "=== r1 on ALL 75 training tasks x1, 1800 s, shard $SHARD ($(echo $TASKS | wc -w) tasks), GPU $GPU $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
GPU=$GPU GATE=0 MAXTOK=2048 TBTIMEOUT=64800 CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_clean NAME=train75_r1_1800_$SHARD PORT=$PORT ATT=1 MINFREE=40000 DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=1800 TASKS="$TASKS" \
  bash eval_gated.sh > "$TMP/train75_r1_1800_$SHARD.log" 2>&1
python3 -I tb_progress.py "$OUT" train75_r1_1800_$SHARD | tee -a "$REP"
echo "train75_r1_1800_$SHARD done $(date '+%m-%d %H:%M')" | tee -a "$REP"
