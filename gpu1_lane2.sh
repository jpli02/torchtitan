#!/bin/bash
# GPU 1 lane, continued (replaces the tail of gpu1_lane.sh, with protection against a harness that is killed from outside: on
# 2026-11-14 14:20 both running harnesses were terminated mid-run and their chains moved on with 12 and 75 results).
#   1. wait for the epoch 8 run that gpu1_lane.sh started (all 75 training tasks);
#   2. run whatever tasks epoch 8 is still missing, until all 75 have a result;
#   3. the same for epoch 6 (13 tasks were finished before it was paused);
#   4. the 11-task test evaluations of epochs 8, 7, 6, 5.
# Start detached:  setsid nohup bash gpu1_lane2.sh > $TMP/gpu1_lane2.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
OUT=/srv/scratch/jli199/tb_runs
DUMP=/home/jli199/boptim_scratch/ouro_tvary_r1_8ep
cd "$WT" || exit 1
REP=$TMP/train_tvary_r1_8ep.txt
count() { python3 -I tb_counts_merged.py "$OUT" "$1" | grep -o 'results=[0-9]*' | cut -d= -f2; }
finish() {  # NAME EPOCH PORT: resume until all 75 tasks have a result (at most four rounds)
  local name=$1 k=$2 port=$3 try
  for try in 1 2 3 4; do
    if [ "$(count "$name")" -ge 75 ]; then echo "[lane] $name complete: $(python3 -I tb_counts_merged.py "$OUT" "$name")" | tee -a "$REP"; return 0; fi
    python3 -I tb_merge_resume.py "$OUT" "$name" | tee -a "$REP"
    NAME=$name CKPT=${DUMP}_ep${k}_clean GPU=1 PORT=$port bash resume_tb_run2.sh
  done
  echo "[lane] $name still incomplete after four rounds: $(python3 -I tb_counts_merged.py "$OUT" "$name")" | tee -a "$REP"
}
until grep -q '\[gpu 1\] train75 chain2 for epochs \[8\] finished' "$REP" 2>/dev/null; do sleep 30; done
echo "[lane] epoch 8 first pass ended $(date '+%m-%d %H:%M')" | tee -a "$REP"
finish train75_ep8_1800 8 8198
finish train75_ep6_1800 6 8196
echo "[lane] training-set runs done $(date '+%m-%d %H:%M'); test evaluations start" | tee -a "$REP"
GPU=1 MINFREE=38000 SKIP_SUBSET=1 EPOCHS_LIST="8 7 6 5" bash epoch_eval_chain.sh
echo "[lane] test evaluations returned $? $(date '+%m-%d %H:%M')" | tee -a "$REP"
