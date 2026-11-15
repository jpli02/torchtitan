#!/bin/bash
# GPU 1 lane (the one whole card), in order of information value:
#   1. epoch 8 on all 75 training tasks (end of the schedule, lowest loss; its memory-starved partial on GPU 0 solved 8 of 12
#      tasks where epochs 4 and 7 solved 4), started as soon as the epoch 5/6 chain lets go of the card;
#   2. the tasks epoch 6 has not run yet (its finished results stay where they are; leftovers go to the resume root);
#   3. the 11-task test evaluations of epochs 8, 7, 6, 5.
# Start detached:  setsid nohup bash gpu1_lane.sh > $TMP/gpu1_lane.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
REP=$TMP/train_tvary_r1_8ep.txt
until grep -q 'train75 chain2 for epochs \[5 6\] finished' "$REP" 2>/dev/null; do sleep 20; done
echo "[lane] card released; epoch 8 starts $(date '+%m-%d %H:%M')" | tee -a "$REP"
GPU=1 MINFREE=38000 EPOCHS_LIST="8" bash epoch_train75_chain2.sh
echo "[lane] epoch 8 chain returned $? $(date '+%m-%d %H:%M')" | tee -a "$REP"
NAME=train75_ep6_1800 CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_8ep_ep6_clean GPU=1 PORT=8196 bash resume_tb_run2.sh
echo "[lane] epoch 6 leftovers returned $? $(date '+%m-%d %H:%M')" | tee -a "$REP"
GPU=1 MINFREE=38000 SKIP_SUBSET=1 EPOCHS_LIST="8 7 6 5" bash epoch_eval_chain.sh
echo "[lane] test evaluations returned $? $(date '+%m-%d %H:%M')" | tee -a "$REP"
