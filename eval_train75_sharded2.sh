#!/bin/bash
# Pass rate of a checkpoint on the 75 take-turns r1 training tasks at the standard 1,800 s budget, in NSHARD shards.
# Same as eval_train75_sharded.sh plus MINFREE / SEQCAP knobs for cards another user half-holds (24 GB free):
#   CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_clean TAG=r1 SHARD=2 NSHARD=4 GPU=2 MINFREE=20000 bash eval_train75_sharded2.sh
set -u
: "${CKPT:?}"; : "${TAG:?}"; : "${SHARD:?}"; : "${NSHARD:?}"; : "${GPU:?}"
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
POOL=/home/jli199/boptim_scratch/tb_variants/pool
OUT=/srv/scratch/jli199/tb_runs
REP=$TMP/eval_train_tasks.txt
cd "$WT" || exit 1
TASKS=$(cat "$TMP/train_tasks_r1.txt" | tr ' ' '\n' | awk -v s="$SHARD" -v n="$NSHARD" 'NF && (NR-1)%n==s-1' | tr '\n' ' ')
NAME=train75_${TAG}_1800_s$SHARD
export OURO_SEQ_CAP=${SEQCAP:-12000}
echo "=== $TAG on the 75 training tasks x1, 1800 s, shard $SHARD/$NSHARD ($(echo $TASKS | wc -w) tasks), GPU $GPU, seq cap $OURO_SEQ_CAP $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
GPU=$GPU GATE=0 MAXTOK=2048 TBTIMEOUT=64800 CKPT=$CKPT NAME=$NAME PORT=$((8190 + SHARD)) ATT=1 MINFREE=${MINFREE:-40000} DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=1800 TASKS="$TASKS" \
  bash eval_gated.sh > "$TMP/$NAME.log" 2>&1
python3 -I tb_progress.py "$OUT" $NAME | tee -a "$REP"
echo "$NAME done $(date '+%m-%d %H:%M')" | tee -a "$REP"
