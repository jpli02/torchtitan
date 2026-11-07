#!/bin/bash
# Speed-as-budget test: r1 on the 22 ep4-solved training tasks with TRUE decode-time early exit (THR, default 0.3)
# served from ouro_tvary_r1_earlyexit (r1 weights + the kvshare modeling file). Same protocol as eval_train22_r1.sh
# (temperature 0.7, 1,800 s, 2,048-token turn cap): if timeouts drop and pass@1 rises, wall-clock per turn is the lever.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
POOL=/home/jli199/boptim_scratch/tb_variants/pool
OUT=/srv/scratch/jli199/tb_runs
THR=${THR:-0.3}
TAG=${TAG:-ee03}
TASKS=$(cat "$TMP/train_tasks_ep4_resolved.txt")
REP=$TMP/eval_train_tasks.txt
cd "$WT" || exit 1
export OURO_SEQ_CAP=12000
echo "=== r1 TRUE EARLY EXIT q=$THR on the 22 ep4-solved training tasks x1, 1800 s, GPU ${GPU:-5} $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
GPU=${GPU:-5} GATE=0 THR=$THR MAXTOK=2048 CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_earlyexit NAME=train22_r1_$TAG PORT=${PORT:-8167} ATT=1 MINFREE=40000 DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=1800 TASKS="$TASKS" \
  bash eval_gated.sh > "$TMP/train22_r1_$TAG.log" 2>&1
python3 -I tb_progress.py "$OUT" train22_r1_$TAG | tee -a "$REP"
python3 -I tb_traj_stats.py "$OUT" train22_r1_$TAG | grep -E 'episodes|context' | tee -a "$REP"
grep -oE 'secs=[0-9.]+ tok/s=[0-9.]+' /home/jli199/terminal_bench_eval/logs/train22_r1_${TAG}_server.log 2>/dev/null | awk -F'[= ]' '{s+=$2; t+=$4; n++} END {if(n) printf "   server: %d turns, %.1f s/turn, %.1f tok/s\n", n, s/n, t/n}' | tee -a "$REP"
echo "train22_r1_$TAG done $(date '+%m-%d %H:%M')" | tee -a "$REP"
