#!/bin/bash
# No-reasoning multiple-choice regime: the assistant turn is forced to start with "\boxed{" so the next token is the
# answer letter (the Ouro paper's own efficiency curves are single-token MMLU answers). Minutes per point.
#   NAME=supergpqa_direct DATA=$TMP/supergpqa.jsonl GPU=4 bash mc_direct.sh     -> $TMP/bench_${NAME}.txt
# Points: K=4, K=2, K=1, gate q=0.5, q=0.3, q=0.1, per-step (0.1,0.5,0.5), (0.3,0.5,0.5); gather mode (loops counted).
set -u
: "${NAME:?}"; : "${DATA:?}"; : "${GPU:?}"
N=${N:-300}; BATCH=${BATCH:-16}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
REP=$TMP/bench_${NAME}.txt
cd "$WT" || exit 1
run() {  # run <label> <extra args...>
  local label=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task mc \
    --n "$N" --batch "$BATCH" --max_new 8 --assistant_prefix '\\boxed{' "$@" --out "$TMP/bench_${NAME}_${label}.jsonl" > "$TMP/bench_${NAME}_${label}.log" 2>&1
  grep '^RESULT' "$TMP/bench_${NAME}_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP" | sed 's/"ckpt": "[^"]*", //' | cut -c1-200
  grep -q '^RESULT' "$TMP/bench_${NAME}_${label}.log" || echo "$label FAILED: $(grep -E 'Error' "$TMP/bench_${NAME}_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"
}
echo "=== MC DIRECT $NAME (n=$N, forced boxed letter, gpu $GPU) start $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
run k4 --steps 4
run k2 --steps 2
run k1 --steps 1
run q0.5 --steps 4 --threshold 0.5
run q0.3 --steps 4 --threshold 0.3
run q0.1 --steps 4 --threshold 0.1
run steps0.1-0.5-0.5 --steps 4 --threshold_steps 0.1,0.5,0.5
run steps0.3-0.5-0.5 --steps 4 --threshold_steps 0.3,0.5,0.5
echo "=== MC DIRECT $NAME done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
