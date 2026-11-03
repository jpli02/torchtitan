#!/bin/bash
# LiveCodeBench (direct mode) under TRUE decode-time early exit, batch 1, first N problems of lcb_all.jsonl:
# full depth | gate q=0.5 | gate q=0.3 | gate q=0.7, each graded against the official tests (docker python:3.12-slim).
#   GPU=3 N=120 bash wallclock_lcb.sh      -> report $TMP/wallclock_lcb.txt
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
DATA=$TMP/lcb_all.jsonl
GPU=${GPU:-3}; N=${N:-120}; MAXNEW=${MAXNEW:-1024}
S=/srv/scratch/jli199/lcb_grade_stage; mkdir -p "$S"; chmod 777 "$S"
REP=$TMP/wallclock_lcb.txt
cd "$WT" || exit 1
[ -f "$S/$(basename "$DATA")" ] || cp "$DATA" "$S"/
cmp -s "$WT/lcb_grade.py" "$S/lcb_grade.py" || cp "$WT/lcb_grade.py" "$S"/
run() {  # run <label> <extra args...>
  local label=$1; shift
  echo "--- $label start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task lcb --steps 4 \
    --n "$N" --batch 1 --max_new "$MAXNEW" --assistant_prefix '```python\n' "$@" --out "$TMP/wcl_${label}.jsonl" > "$TMP/wcl_${label}.log" 2>&1
  grep '^RESULT' "$TMP/wcl_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
  grep -q '^RESULT' "$TMP/wcl_${label}.log" || { echo "$label FAILED: $(grep -E 'Error' "$TMP/wcl_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"; return 1; }
  cp "$TMP/wcl_${label}.jsonl" "$S"/
  docker run --rm --network none --memory 12g --cpus 4 -v "$S":/w python:3.12-slim \
    python /w/lcb_grade.py "/w/$(basename "$DATA")" "/w/wcl_${label}.jsonl" "/w/wcl_${label}.graded.json" 2>&1 \
    | grep -E '^GRADED' | sed "s/^/$label /" | tee -a "$REP"
  cp "$S/wcl_${label}.graded.json" "$TMP/" 2>/dev/null
}
echo "=== LCB TRUE-EXIT wall-clock (direct mode, batch 1, n=$N, max_new=$MAXNEW, gpu $GPU) $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
run full
run gate0.5 --cfg decode_early_exit_threshold=0.5
run gate0.3 --cfg decode_early_exit_threshold=0.3
run gate0.7 --cfg decode_early_exit_threshold=0.7
echo "=== LCB TRUE-EXIT done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
