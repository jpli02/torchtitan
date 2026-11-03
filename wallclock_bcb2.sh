#!/bin/bash
# BigCodeBench (direct mode) under TRUE decode-time early exit, batch 1, with selectable points and an item
# OFFSET (labels <point>_o<OFFSET>), appending to the same report as wallclock_bcb.sh:
#   GPU=2 OFFSET=150 N=150 POINTS="full gate0.5" bash wallclock_bcb2.sh     -> $TMP/wallclock_bcb.txt
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
DATA=$TMP/bcb_full300.jsonl
GPU=${GPU:-2}; N=${N:-150}; MAXNEW=${MAXNEW:-768}
OFFSET=${OFFSET:-0}; SUF=""; [ "$OFFSET" != 0 ] && SUF="_o${OFFSET}"
POINTS=${POINTS:-"full gate0.5 gate0.3 gate0.7"}
S=/tmp/bcb_grade_stage; mkdir -p "$S"; chmod 777 "$S"
REP=$TMP/wallclock_bcb.txt
cd "$WT" || exit 1
run() {  # run <point> <extra args...>
  local label=${1}${SUF}; shift
  echo "--- $label start $(date '+%m-%d %H:%M') (gpu $GPU, offset $OFFSET)" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task bcb --steps 4 \
    --n "$N" --batch 1 --max_new "$MAXNEW" --seed_offset "$OFFSET" --assistant_prefix '```python\n' "$@" --out "$TMP/wcb_${label}.jsonl" > "$TMP/wcb_${label}.log" 2>&1
  grep '^RESULT' "$TMP/wcb_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
  grep -q '^RESULT' "$TMP/wcb_${label}.log" || { echo "$label FAILED: $(grep -E 'Error' "$TMP/wcb_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"; return 1; }
  cmp -s "$WT/bcb_grade.py" "$S/bcb_grade.py" || cp "$WT/bcb_grade.py" "$S"/
  [ -f "$S/bcb_full300.jsonl" ] || cp "$DATA" "$S"/
  cp "$TMP/wcb_${label}.jsonl" "$S"/
  docker run --rm --network none --memory 6g --cpus 4 --user "$(id -u):$(id -g)" -v "$S":/w --entrypoint python \
    bigcodebench/bigcodebench-evaluate:latest /w/bcb_grade.py /w/bcb_full300.jsonl "/w/wcb_${label}.jsonl" "/w/wcb_${label}.graded.json" 2>&1 \
    | grep -E '^GRADED' | sed "s/^/$label /" | tee -a "$REP"
  cp "$S/wcb_${label}.graded.json" "$TMP/" 2>/dev/null
}
for p in $POINTS; do
  case "$p" in
    full) run full ;;
    gate0.7) run gate0.7 --cfg decode_early_exit_threshold=0.7 ;;
    gate0.5) run gate0.5 --cfg decode_early_exit_threshold=0.5 ;;
    gate0.3) run gate0.3 --cfg decode_early_exit_threshold=0.3 ;;
    steps0.3-0.5-0.5) run steps0.3-0.5-0.5 --cfg "decode_early_exit_threshold=[0.3,0.5,0.5,1.0]" ;;
    *) echo "unknown point $p" | tee -a "$REP" ;;
  esac
done
echo "=== BCB TRUE-EXIT points [$POINTS] offset $OFFSET on gpu $GPU done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
