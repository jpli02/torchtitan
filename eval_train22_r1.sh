#!/bin/bash
# Fit check for the take-turns r1 checkpoint at the FULL 1,800 s agent budget (the 900 s train-task protocol
# timed out 26 of its first 29 trials because r1 writes ~2x longer turns): the 22 training tasks that the fully
# fit ep4 checkpoint solved at 900 s. Same pool dataset / local output root as eval_train_tasks.sh.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
POOL=/home/jli199/boptim_scratch/tb_variants/pool
OUT=/srv/scratch/jli199/tb_runs
TASKS=$(cat "$TMP/train_tasks_ep4_resolved.txt")
REP=$TMP/eval_train_tasks.txt
cd "$WT" || exit 1
export OURO_SEQ_CAP=12000
echo "=== r1 on the 22 ep4-solved training tasks x1, 1800 s, GPU ${GPU:-4} $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
GPU=${GPU:-4} GATE=0 CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_clean NAME=train22_r1_1800 PORT=8164 ATT=1 MINFREE=40000 DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=1800 TASKS="$TASKS" \
  bash eval_gated.sh > "$TMP/train22_r1_1800.log" 2>&1
python3 -I tb_progress.py "$OUT" train22_r1_1800 | tee -a "$REP"
grep -oE 'secs=[0-9.]+ tok/s=[0-9.]+' /home/jli199/terminal_bench_eval/logs/train22_r1_1800_server.log 2>/dev/null | awk -F'[= ]' '{s+=$2; t+=$4; n++} END {if(n) printf "   server: %d turns, %.1f s/turn, %.1f tok/s\n", n, s/n, t/n}' | tee -a "$REP"
echo "train22_r1_1800 done $(date '+%m-%d %H:%M')" | tee -a "$REP"
