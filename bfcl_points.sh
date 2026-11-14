#!/bin/bash
# BFCL v3 single-turn AST categories (ICML 2025) on Ouro-2.6B: depth / gate points, graded by the official AST checker inside the
# sandbox image bfcl-grader. Gather mode, loops counted. Two regimes:
#   plain   the official prompt as is (the model may reason before the call list), 512-token budget   -> comparable to the published
#           base Ouro-2.6B number (47.9 % non-live, arXiv 2608.18171)
#   direct  the answer is forced to start with "[" (the call list), 256-token budget
#   GPU=0 bash bfcl_points.sh                 # 400-item sample (100 per leaderboard category)
#   GPU=0 DATA=bfcl_all N=1150 bash bfcl_points.sh
# Report: $TMP/bench_bfcl.txt  (RESULT <label> {...} and GRADED <label> n=.. overall=.. micro=.. per-category .. loops=..)
set -u
: "${GPU:?}"
DATA=${DATA:-bfcl400}; N=${N:-400}; BATCH=${BATCH:-4}; NAME=${NAME:-bfcl}   # NAME=bfcl_smoke keeps a test run out of the report
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
STAGE=/srv/scratch/jli199/bfcl_stage
REP=$TMP/bench_${NAME}.txt
cd "$WT" || exit 1
cp "$WT/bfcl_tool.py" "$STAGE/"
[ -f "$TMP/$DATA.jsonl" ] || cp "$STAGE/$DATA.jsonl" "$TMP/"
run() {  # run <label> <max_new> <prefix or ""> <extra depth_eval args...>
  local label=$1 maxnew=$2 prefix=$3; shift 3
  local pre=(); [ -n "$prefix" ] && pre=(--assistant_prefix "$prefix")
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$TMP/$DATA.jsonl" --task bfcl \
    --n "$N" --batch "$BATCH" --max_new "$maxnew" "${pre[@]}" "$@" --out "$TMP/bench_${NAME}_${label}.jsonl" > "$TMP/bench_${NAME}_${label}.log" 2>&1
  grep '^RESULT' "$TMP/bench_${NAME}_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP" | sed 's/"ckpt": "[^"]*", //' | cut -c1-200
  if ! grep -q '^RESULT' "$TMP/bench_${NAME}_${label}.log"; then
    echo "$label FAILED: $(grep -E 'Error' "$TMP/bench_${NAME}_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"; return
  fi
  cp "$TMP/bench_${NAME}_${label}.jsonl" "$STAGE/"
  docker run --rm --network none --memory 8g --cpus 4 -v "$STAGE":/w bfcl-grader python /w/bfcl_tool.py grade "/w/$DATA.jsonl" \
    "/w/bench_${NAME}_${label}.jsonl" "/w/bench_${NAME}_${label}.graded.json" 2>&1 | grep '^GRADED' | sed "s/^GRADED /GRADED $label /" | tee -a "$REP"
  cp "$STAGE/bench_${NAME}_${label}.graded.json" "$TMP/" 2>/dev/null
}
echo "=== BFCL $DATA (n=$N, batch $BATCH, gpu $GPU) start $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
for pt in ${POINTS:-plain_k4 direct_k4 direct_k2 direct_k1 direct_q0.3}; do
  case $pt in
    plain_k4) run plain_k4 512 "" --steps 4 ;;
    plain_q0.3) run plain_q0.3 512 "" --steps 4 --threshold 0.3 ;;
    direct_k4) run direct_k4 256 "[" --steps 4 ;;
    direct_k2) run direct_k2 256 "[" --steps 2 ;;
    direct_k1) run direct_k1 256 "[" --steps 1 ;;
    direct_q0.5) run direct_q0.5 256 "[" --steps 4 --threshold 0.5 ;;
    direct_q0.3) run direct_q0.3 256 "[" --steps 4 --threshold 0.3 ;;
    direct_steps0.3-0.5-0.5) run direct_steps0.3-0.5-0.5 256 "[" --steps 4 --threshold_steps 0.3,0.5,0.5 ;;
    *) echo "unknown point $pt" | tee -a "$REP" ;;
  esac
done
echo "=== BFCL $DATA done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
