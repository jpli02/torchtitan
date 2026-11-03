#!/bin/bash
# One LiveCodeBench direct-mode point: generate (depth_eval --task lcb, forced ```python prefix) and grade
# (lcb_grade.py in docker python:3.12-slim). Appends to the same report as lcb_pipeline.sh.
#   NAME=lcb342 DATA=$TMP/lcb_all.jsonl GPU=3 bash lcb_point.sh q0.5 --steps 4 --threshold 0.5
set -u
: "${NAME:?}"; : "${DATA:?}"; : "${GPU:?}"
N=${N:-342}; MAXNEW=${MAXNEW:-1024}; BATCH=${BATCH:-6}
EXTRA=${EXTRA:-"--assistant_prefix \`\`\`python\\n"}
label=$1; shift
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
S=/srv/scratch/jli199/lcb_grade_stage; mkdir -p "$S"; chmod 777 "$S"
REP=$TMP/bench_${NAME}.txt
cd "$WT" || exit 1
[ -f "$S/$(basename "$DATA")" ] || cp "$DATA" "$S"/
cmp -s "$WT/lcb_grade.py" "$S/lcb_grade.py" || cp "$WT/lcb_grade.py" "$S"/
echo "--- $label start $(date '+%m-%d %H:%M') (gpu $GPU)" | tee -a "$REP"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task lcb \
  --n "$N" --batch "$BATCH" --max_new "$MAXNEW" $EXTRA "$@" --out "$TMP/bench_${NAME}_${label}.jsonl" > "$TMP/bench_${NAME}_${label}.log" 2>&1
grep '^RESULT' "$TMP/bench_${NAME}_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
grep -q '^RESULT' "$TMP/bench_${NAME}_${label}.log" || { echo "$label FAILED: $(grep -E 'Error' "$TMP/bench_${NAME}_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"; exit 1; }
cp "$TMP/bench_${NAME}_${label}.jsonl" "$S"/
docker run --rm --network none --memory 12g --cpus 4 -v "$S":/w python:3.12-slim \
  python /w/lcb_grade.py "/w/$(basename "$DATA")" "/w/bench_${NAME}_${label}.jsonl" "/w/bench_${NAME}_${label}.graded.json" 2>&1 \
  | grep -E '^GRADED' | sed "s/^/$label /" | tee -a "$REP"
cp "$S/bench_${NAME}_${label}.graded.json" "$TMP/" 2>/dev/null
