#!/bin/bash
# IFBench (NeurIPS 2025 D&B, 299 prompts, 58 out-of-distribution constraints) depth/gate points on one GPU, graded with the
# official verifiers inside the sandboxed image ifbench-grader (boptim_lcb/ifbench_grader.Dockerfile).
#   GPU=1 VARIANT=think bash ifb_points.sh      # thinking regime, 2,048-token budget (the reasoning block is stripped)
#   GPU=1 VARIANT=direct bash ifb_points.sh     # an empty think block is forced -> the answer directly, 1,024 tokens
# Points (gather mode, loops counted), POINTS="k4 k2 q0.3 q0.5 steps0.3-0.5-0.5" by default.  Report: $TMP/bench_ifbench_$VARIANT.txt
set -u
: "${GPU:?}"
VARIANT=${VARIANT:-think}
N=${N:-299}; BATCH=${BATCH:-8}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
STAGE=/srv/scratch/jli199/ifb_grade_stage
NAME=ifbench_$VARIANT
REP=$TMP/bench_${NAME}.txt
if [ "$VARIANT" = direct ]; then EXTRA=(--assistant_prefix '<think>\n\n</think>\n\n'); MAXNEW=${MAXNEW:-1024}; else EXTRA=(--assistant_prefix '<think>\n'); MAXNEW=${MAXNEW:-2048}; fi
mkdir -p "$STAGE"; chmod 777 "$STAGE"; cp "$TMP/ifbench.jsonl" "$STAGE/"; cp "$WT/ifb_grade.py" "$STAGE/"
cd "$WT" || exit 1
run() {  # run <label> <extra depth_eval args...>
  local label=$1; shift
  echo "--- $label start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$TMP/ifbench.jsonl" --task ifb \
    --n "$N" --batch "$BATCH" --max_new "$MAXNEW" "${EXTRA[@]}" "$@" --out "$TMP/bench_${NAME}_${label}.jsonl" > "$TMP/bench_${NAME}_${label}.log" 2>&1
  grep '^RESULT' "$TMP/bench_${NAME}_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
  if ! grep -q '^RESULT' "$TMP/bench_${NAME}_${label}.log"; then
    echo "$label FAILED: $(grep -E 'Error' "$TMP/bench_${NAME}_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"
    return
  fi
  cp "$TMP/bench_${NAME}_${label}.jsonl" "$STAGE/"
  docker run --rm --network none --memory 8g --cpus 4 -v "$STAGE":/w ifbench-grader python /w/ifb_grade.py /w/ifbench.jsonl \
    "/w/bench_${NAME}_${label}.jsonl" "/w/bench_${NAME}_${label}.graded.json" 2>&1 | grep '^GRADED' | sed "s/^GRADED /GRADED $label /" | tee -a "$REP"
  cp "$STAGE/bench_${NAME}_${label}.graded.json" "$TMP/" 2>/dev/null
}
echo "=== IFBENCH $VARIANT (n=$N, max_new $MAXNEW, batch $BATCH, gpu $GPU) start $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
for pt in ${POINTS:-k4 k2 q0.3 q0.5 steps0.3-0.5-0.5}; do
  case $pt in
    k4) run k4 --steps 4 ;;
    k2) run k2 --steps 2 ;;
    q0.3) run q0.3 --steps 4 --threshold 0.3 ;;
    q0.5) run q0.5 --steps 4 --threshold 0.5 ;;
    steps0.3-0.5-0.5) run steps0.3-0.5-0.5 --steps 4 --threshold_steps 0.3,0.5,0.5 ;;
    *) echo "unknown point $pt" | tee -a "$REP" ;;
  esac
done
echo "=== IFBENCH $VARIANT done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
