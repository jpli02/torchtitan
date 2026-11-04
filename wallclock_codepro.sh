#!/bin/bash
# HumanEval Pro + MBPP Pro (direct mode) under TRUE decode-time early exit, batch 1, first N problems of codepro.jsonl
# (shuffle-free: the file is HumanEval Pro 164 then MBPP Pro 378, so OFFSET selects the source), graded with
# codepro_grade.py in docker python:3.12-slim.
#   GPU=1 N=150 POINTS="full gate0.5 gate0.3" bash wallclock_codepro.sh   -> $TMP/wallclock_codepro.txt
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
DATA=$TMP/codepro.jsonl
GPU=${GPU:-1}; N=${N:-150}; MAXNEW=${MAXNEW:-1024}
OFFSET=${OFFSET:-0}; SUF=""; [ "$OFFSET" != 0 ] && SUF="_o${OFFSET}"
POINTS=${POINTS:-"full gate0.5 gate0.3 gate0.7"}
S=/srv/scratch/jli199/codepro_grade_stage; mkdir -p "$S"; chmod 777 "$S"
REP=$TMP/wallclock_codepro.txt
cd "$WT" || exit 1
cp "$DATA" "$S"/
cmp -s "$WT/codepro_grade.py" "$S/codepro_grade.py" || cp "$WT/codepro_grade.py" "$S"/
run() {  # run <point> <extra args...>
  local label=${1}${SUF}; shift
  echo "--- $label start $(date '+%m-%d %H:%M') (gpu $GPU, offset $OFFSET)" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task codepro --steps 4 \
    --n "$N" --batch 1 --max_new "$MAXNEW" --seed_offset "$OFFSET" --assistant_prefix '```python\n' "$@" --out "$TMP/wcp_${label}.jsonl" > "$TMP/wcp_${label}.log" 2>&1
  grep '^RESULT' "$TMP/wcp_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
  grep -q '^RESULT' "$TMP/wcp_${label}.log" || { echo "$label FAILED: $(grep -E 'Error' "$TMP/wcp_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"; return 1; }
  cp "$TMP/wcp_${label}.jsonl" "$S"/
  docker run --rm --network none --memory 4g --cpus 4 -v "$S":/w python:3.12-slim \
    python /w/codepro_grade.py /w/codepro.jsonl "/w/wcp_${label}.jsonl" "/w/wcp_${label}.graded.json" 2>&1 \
    | grep -E '^GRADED' | sed "s/^/$label /" | tee -a "$REP"
  cp "$S/wcp_${label}.graded.json" "$TMP/" 2>/dev/null
}
for p in $POINTS; do
  case "$p" in
    full) run full ;;
    gate0.7) run gate0.7 --cfg decode_early_exit_threshold=0.7 ;;
    gate0.5) run gate0.5 --cfg decode_early_exit_threshold=0.5 ;;
    gate0.3) run gate0.3 --cfg decode_early_exit_threshold=0.3 ;;
    gate0.2) run gate0.2 --cfg decode_early_exit_threshold=0.2 ;;
    *) echo "unknown point $p" | tee -a "$REP" ;;
  esac
done
echo "=== CODEPRO TRUE-EXIT points [$POINTS] offset $OFFSET on gpu $GPU done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
