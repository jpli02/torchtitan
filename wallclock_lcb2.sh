#!/bin/bash
# Same as wallclock_lcb.sh (LiveCodeBench direct mode under TRUE decode-time early exit, batch 1, first N
# problems, graded) but the points are selectable, so a second card can take part of the chain:
#   GPU=5 POINTS="gate0.7 gate0.3" bash wallclock_lcb2.sh      -> appends to $TMP/wallclock_lcb.txt
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
DATA=$TMP/lcb_all.jsonl
GPU=${GPU:-5}; N=${N:-120}; MAXNEW=${MAXNEW:-1024}
POINTS=${POINTS:-"full gate0.5 gate0.3 gate0.7"}
S=/srv/scratch/jli199/lcb_grade_stage; mkdir -p "$S"; chmod 777 "$S"
REP=$TMP/wallclock_lcb.txt
cd "$WT" || exit 1
[ -f "$S/$(basename "$DATA")" ] || cp "$DATA" "$S"/
cmp -s "$WT/lcb_grade.py" "$S/lcb_grade.py" || cp "$WT/lcb_grade.py" "$S"/
run() {  # run <label> <extra args...>
  local label=$1; shift
  echo "--- $label start $(date '+%m-%d %H:%M') (gpu $GPU)" | tee -a "$REP"
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
for p in $POINTS; do
  case "$p" in
    full) run full ;;
    gate0.5) run gate0.5 --cfg decode_early_exit_threshold=0.5 ;;
    gate0.3) run gate0.3 --cfg decode_early_exit_threshold=0.3 ;;
    gate0.7) run gate0.7 --cfg decode_early_exit_threshold=0.7 ;;
    *) echo "unknown point $p" | tee -a "$REP" ;;
  esac
done
echo "=== LCB TRUE-EXIT points [$POINTS] on gpu $GPU done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
