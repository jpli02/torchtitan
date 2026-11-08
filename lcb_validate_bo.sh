#!/bin/bash
# Held-out validation of BO-found routers on LiveCodeBench problems 120..341 (the BO searched on 0..119), under TRUE decode-time early
# exit, batch 1, direct mode, graded in the sandbox; same protocol and labels (*_o120 over 120..239, *_o240 over 240..341) as
# wallclock_lcb3.sh so the page can pool the two slices.
#   GPU=0 CANDS="bo20:0.529,0.058,0.348 bo15:0.408,0.269,0.313" bash lcb_validate_bo.sh      -> appends to $TMP/wallclock_lcb_bo.txt
#   a plain gate is also accepted:  CANDS="gate0.3:0.3"
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
DATA=$TMP/lcb_all.jsonl
GPU=${GPU:-0}; MAXNEW=${MAXNEW:-1024}
: "${CANDS:?}"
OFFSETS=${OFFSETS:-"120 240"}
S=/srv/scratch/jli199/lcb_grade_stage; mkdir -p "$S"; chmod 777 "$S"
REP=$TMP/wallclock_lcb_bo.txt
cd "$WT" || exit 1
[ -f "$S/$(basename "$DATA")" ] || cp "$DATA" "$S"/
cmp -s "$WT/lcb_grade.py" "$S/lcb_grade.py" || cp "$WT/lcb_grade.py" "$S"/
run() {  # run <label> <offset> <n> <cfg value>
  local label=$1 offset=$2 n=$3 cfg=$4
  echo "--- $label start $(date '+%m-%d %H:%M') (gpu $GPU, offset $offset, n $n, cfg $cfg)" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task lcb --steps 4 \
    --n "$n" --batch 1 --max_new "$MAXNEW" --seed_offset "$offset" --assistant_prefix '```python\n' --cfg "decode_early_exit_threshold=$cfg" \
    --out "$TMP/wcl_${label}.jsonl" > "$TMP/wcl_${label}.log" 2>&1
  grep '^RESULT' "$TMP/wcl_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
  grep -q '^RESULT' "$TMP/wcl_${label}.log" || { echo "$label FAILED: $(grep -E 'Error' "$TMP/wcl_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"; return 1; }
  cp "$TMP/wcl_${label}.jsonl" "$S"/
  docker run --rm --network none --memory 12g --cpus 4 -v "$S":/w python:3.12-slim \
    python /w/lcb_grade.py "/w/$(basename "$DATA")" "/w/wcl_${label}.jsonl" "/w/wcl_${label}.graded.json" 2>&1 \
    | grep -E '^GRADED' | sed "s/^/$label /" | tee -a "$REP"
  cp "$S/wcl_${label}.graded.json" "$TMP/" 2>/dev/null
}
echo "=== LCB held-out validation [$CANDS] on gpu $GPU start $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
for cand in $CANDS; do
  name=${cand%%:*}; thr=${cand#*:}
  case "$thr" in
    *,*) cfg="[$thr,1.0]" ;;   # per-step thresholds q1,q2,q3 (loop 4 always exits)
    *) cfg="$thr" ;;           # one gate threshold
  esac
  for off in $OFFSETS; do
    n=120; [ "$off" = 240 ] && n=102
    run "${name}_o${off}" "$off" "$n" "$cfg"
  done
done
echo "=== LCB held-out validation [$CANDS] on gpu $GPU done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
