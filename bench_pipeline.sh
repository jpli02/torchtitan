#!/bin/bash
# Full router-headroom pipeline for one benchmark, on one GPU:
#   1. K=4 full-depth generation (baseline accuracy)      2. oracle bound + per-token features from it
#   3. proxy frontier (ms)                                 4. K=2 generation (depth sensitivity)
#   5. released gate q=0.3                                 6. per-step rule (0.1,0.5,0.5)
#   NAME=mmlupro TASK=mc DATA=$TMP/mmlupro.jsonl N=300 GPU=4 MAXNEW=2048 bash bench_pipeline.sh
set -u
: "${NAME:?}"; : "${TASK:?}"; : "${DATA:?}"; : "${GPU:?}"
N=${N:-300}; BATCH=${BATCH:-6}; MAXNEW=${MAXNEW:-2048}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
REP=$TMP/bench_${NAME}.txt
cd "$WT" || exit 1
run() {  # run <label> <extra depth_eval args...>
  local label=$1; shift
  echo "--- $label start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task "$TASK" \
    --n "$N" --batch "$BATCH" --max_new "$MAXNEW" "$@" --out "$TMP/bench_${NAME}_${label}.jsonl" > "$TMP/bench_${NAME}_${label}.log" 2>&1
  grep '^RESULT' "$TMP/bench_${NAME}_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
  grep -q '^RESULT' "$TMP/bench_${NAME}_${label}.log" || echo "$label FAILED: $(grep -E 'Error' "$TMP/bench_${NAME}_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"
}
echo "=== BENCH $NAME ($TASK, n=$N, gpu $GPU) start $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
run k4 --steps 4
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" oracle_depth.py "$CKPT" "$TMP/bench_${NAME}_k4.jsonl" --task "$TASK" --data "$DATA" \
  --out "$TMP/bench_${NAME}_oracle.json" --dump "$TMP/feat_${NAME}_k4.pt" > "$TMP/bench_${NAME}_oracle.log" 2>&1
grep -vE 'Warning|warn|deprecated|^ORACLE|gens,|dumped' "$TMP/bench_${NAME}_oracle.log" | tail -9 | tee -a "$REP"
"$PY" router_proxy.py "$TMP/feat_${NAME}_k4.pt" --out "$TMP/proxy_${NAME}_frontier.json" 2>/dev/null | tail -22 | tee -a "$REP"
run k2 --steps 2
run q0.3 --steps 4 --threshold 0.3
run steps0.1-0.5-0.5 --steps 4 --threshold_steps 0.1,0.5,0.5
echo "=== BENCH $NAME done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
