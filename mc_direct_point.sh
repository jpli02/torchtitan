#!/bin/bash
# One direct-MC point (same protocol and report format as mc_direct.sh), for re-running a point that hit out-of-memory on a shared card.
#   NAME=gpqa_direct DATA=$TMP/gpqa_diamond.jsonl GPU=0 N=198 BATCH=4 LABEL=k4 bash mc_direct_point.sh --steps 4
set -u
: "${NAME:?}"; : "${DATA:?}"; : "${GPU:?}"; : "${LABEL:?}"
N=${N:-300}; BATCH=${BATCH:-4}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
REP=$TMP/bench_${NAME}.txt
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task mc \
  --n "$N" --batch "$BATCH" --max_new 8 --assistant_prefix '\\boxed{' "$@" --out "$TMP/bench_${NAME}_${LABEL}.jsonl" > "$TMP/bench_${NAME}_${LABEL}.log" 2>&1
grep '^RESULT' "$TMP/bench_${NAME}_${LABEL}.log" | sed "s/^RESULT /RESULT $LABEL /" | tee -a "$REP" | sed 's/"ckpt": "[^"]*", //' | cut -c1-200
grep -q '^RESULT' "$TMP/bench_${NAME}_${LABEL}.log" || echo "$LABEL FAILED: $(grep -E 'Error' "$TMP/bench_${NAME}_${LABEL}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"
