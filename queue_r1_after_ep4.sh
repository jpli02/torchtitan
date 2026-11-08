#!/bin/bash
# When the four ep4 training-set shards are done, run r1 on the same 75 tasks at 1,800 s in four shards (GPUs 1,2,4,5).
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
REP=$TMP/eval_train_tasks.txt
END=$(( $(date +%s) + 72000 ))
while [ "$(date +%s)" -lt "$END" ]; do
  n=$(grep -c 'train75_ep4_1800_s[1-4] done' "$REP" 2>/dev/null || echo 0)
  [ "$n" -ge 4 ] && break
  sleep 300
done
cd "$WT" || exit 1
CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_clean
CKPT=$CKPT TAG=r1 SHARD=1 NSHARD=4 GPU=1 bash eval_train75_sharded.sh > "$TMP/queue_r1_s1.log" 2>&1 &
CKPT=$CKPT TAG=r1 SHARD=2 NSHARD=4 GPU=2 bash eval_train75_sharded.sh > "$TMP/queue_r1_s2.log" 2>&1 &
CKPT=$CKPT TAG=r1 SHARD=3 NSHARD=4 GPU=4 bash eval_train75_sharded.sh > "$TMP/queue_r1_s3.log" 2>&1 &
CKPT=$CKPT TAG=r1 SHARD=4 NSHARD=4 GPU=5 bash eval_train75_sharded.sh > "$TMP/queue_r1_s4.log" 2>&1 &
wait
echo "r1 75-task shards finished $(date '+%m-%d %H:%M')" | tee -a "$REP"
