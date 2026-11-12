#!/bin/bash
# Continue a Terminal-Bench evaluation that a harness cap ended, WITHOUT touching its finished results: eval_gated.sh clears
# OUTROOT/NAME on every launch, so the leftover tasks run into a separate root (OUTROOT/resume_NAME/NAME) and counts are merged
# over both roots (tb_counts_merged.py).
#   NAME=train75_ep7_1800 CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_8ep_ep7_clean GPU=5 PORT=8197 bash resume_tb_run2.sh
set -u
: "${NAME:?}"; : "${CKPT:?}"; : "${GPU:?}"
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
POOL=${DATASET:-/home/jli199/boptim_scratch/tb_variants/pool}
OUT=${OUTROOT:-/srv/scratch/jli199/tb_runs}
TASKLIST=${TASKLIST:-$(cat "$TMP/train_tasks_r1.txt")}
REP=${REP:-$TMP/train_tvary_r1_8ep.txt}
cd "$WT" || exit 1
REMAINING=$(python3 -I tb_counts_merged.py "$OUT" "$NAME" --remaining $TASKLIST)
if [ -z "$REMAINING" ]; then echo "$NAME: nothing left to run" | tee -a "$REP"; exit 0; fi
echo "$NAME: resuming $(echo $REMAINING | wc -w) tasks on GPU $GPU into $OUT/resume_$NAME $(date '+%m-%d %H:%M')" | tee -a "$REP"
export OURO_SEQ_CAP=12000
GPU=$GPU GATE=${GATE:-0} MAXTOK=${MAXTOK:-2048} TBTIMEOUT=${TBTIMEOUT:-200000} CKPT=$CKPT NAME=$NAME PORT=${PORT:-8195} ATT=${ATT:-1} MINFREE=${MINFREE:-38000} \
  DATASET=$POOL OUTROOT=$OUT/resume_$NAME AGENT_TIMEOUT=${AGENT_TIMEOUT:-1800} TASKS="$REMAINING" bash eval_gated.sh > "$TMP/${NAME}_resume.log" 2>&1
python3 -I tb_counts_merged.py "$OUT" "$NAME" | tee -a "$REP"
