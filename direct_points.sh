#!/bin/bash
# Depth / gate points for a short-answer benchmark with the reasoning skipped by a forced assistant prefix (gather mode, loops counted):
# K=4, K=2, K=1, gate q=0.5, 0.3, 0.1, per-step (0.3,0.5,0.5) and (0.1,0.5,0.5). Generalises mc_direct.sh to any depth_eval task.
#   NAME=lcbexec_direct TASK=crux DATA=$TMP/lcb_exec.jsonl PREFIX='\\boxed{' MAXNEW=64 N=479 BATCH=8 GPU=3 bash direct_points.sh
#   -> $TMP/bench_${NAME}.txt (RESULT <label> {...} lines), generations in $TMP/bench_${NAME}_<label>.jsonl
set -u
: "${NAME:?}"; : "${TASK:?}"; : "${DATA:?}"; : "${GPU:?}"; : "${PREFIX:?}"
N=${N:-300}; BATCH=${BATCH:-8}; MAXNEW=${MAXNEW:-64}
POINTS=${POINTS:-"k4 k2 k1 q0.5 q0.3 q0.1 steps0.3-0.5-0.5 steps0.1-0.5-0.5"}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
REP=$TMP/bench_${NAME}.txt
cd "$WT" || exit 1
run() {  # run <label> <extra args...>
  local label=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task "$TASK" \
    --n "$N" --batch "$BATCH" --max_new "$MAXNEW" --assistant_prefix "$PREFIX" "$@" --out "$TMP/bench_${NAME}_${label}.jsonl" > "$TMP/bench_${NAME}_${label}.log" 2>&1
  grep '^RESULT' "$TMP/bench_${NAME}_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP" | sed 's/"ckpt": "[^"]*", //' | cut -c1-220
  grep -q '^RESULT' "$TMP/bench_${NAME}_${label}.log" || echo "$label FAILED: $(grep -E 'Error' "$TMP/bench_${NAME}_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"
}
echo "=== DIRECT $NAME (task $TASK, n=$N, batch $BATCH, max_new $MAXNEW, gpu $GPU) start $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
for pt in $POINTS; do
  case $pt in
    k4) run k4 --steps 4 ;;
    k2) run k2 --steps 2 ;;
    k1) run k1 --steps 1 ;;
    q0.5) run q0.5 --steps 4 --threshold 0.5 ;;
    q0.3) run q0.3 --steps 4 --threshold 0.3 ;;
    q0.1) run q0.1 --steps 4 --threshold 0.1 ;;
    steps0.3-0.5-0.5) run steps0.3-0.5-0.5 --steps 4 --threshold_steps 0.3,0.5,0.5 ;;
    steps0.1-0.5-0.5) run steps0.1-0.5-0.5 --steps 4 --threshold_steps 0.1,0.5,0.5 ;;
    *) echo "unknown point $pt" | tee -a "$REP" ;;
  esac
done
echo "=== DIRECT $NAME done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
