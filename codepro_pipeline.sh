#!/bin/bash
# HumanEval Pro + MBPP Pro (542 self-invoking problems, 2025) router-headroom pipeline in DIRECT mode, one GPU;
# each point graded by running the assert tests in docker python:3.12-slim (codepro_grade.py).
#   NAME=codepro DATA=$TMP/codepro.jsonl N=542 GPU=4 bash codepro_pipeline.sh
# Points: K=4 (+oracle/features), K=2, gate q=0.5, gate q=0.3, per-step (0.3,0.5,0.5), K=1. Report: $TMP/bench_${NAME}.txt
set -u
: "${NAME:?}"; : "${DATA:?}"; : "${GPU:?}"
N=${N:-542}; MAXNEW=${MAXNEW:-1024}; BATCH=${BATCH:-8}
EXTRA=${EXTRA:-"--assistant_prefix \`\`\`python\\n"}
POINTS=${POINTS:-"k4 oracle k2 q0.5 q0.3 steps k1"}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
S=/srv/scratch/jli199/codepro_grade_stage; mkdir -p "$S"; chmod 777 "$S"
REP=$TMP/bench_${NAME}.txt
cd "$WT" || exit 1
cp "$DATA" "$S"/
gen_and_grade() {  # gen_and_grade <label> <extra depth_eval args...>
  local label=$1; shift
  echo "--- $label start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task codepro \
    --n "$N" --batch "$BATCH" --max_new "$MAXNEW" $EXTRA "$@" --out "$TMP/bench_${NAME}_${label}.jsonl" > "$TMP/bench_${NAME}_${label}.log" 2>&1
  grep '^RESULT' "$TMP/bench_${NAME}_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
  grep -q '^RESULT' "$TMP/bench_${NAME}_${label}.log" || { echo "$label FAILED: $(grep -E 'Error' "$TMP/bench_${NAME}_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"; return 1; }
  cp "$WT/codepro_grade.py" "$TMP/bench_${NAME}_${label}.jsonl" "$S"/
  docker run --rm --network none --memory 4g --cpus 4 -v "$S":/w python:3.12-slim \
    python /w/codepro_grade.py "/w/$(basename "$DATA")" "/w/bench_${NAME}_${label}.jsonl" "/w/bench_${NAME}_${label}.graded.json" 2>&1 \
    | grep -E '^GRADED' | sed "s/^/$label /" | tee -a "$REP"
  cp "$S/bench_${NAME}_${label}.graded.json" "$TMP/" 2>/dev/null
}
echo "=== CODEPRO $NAME direct (n=$N, max_new=$MAXNEW, batch=$BATCH, gpu $GPU) start $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
for p in $POINTS; do
  case "$p" in
    k4) gen_and_grade k4 --steps 4 ;;
    oracle)
      CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" oracle_depth.py "$CKPT" "$TMP/bench_${NAME}_k4.jsonl" --task codepro --data "$DATA" \
        --out "$TMP/bench_${NAME}_oracle.json" --dump "$TMP/feat_${NAME}_k4.pt" > "$TMP/bench_${NAME}_oracle.log" 2>&1
      grep -vE 'Warning|warn|deprecated|^ORACLE|gens,|dumped' "$TMP/bench_${NAME}_oracle.log" | tail -9 | tee -a "$REP" ;;
    k2) gen_and_grade k2 --steps 2 ;;
    q0.5) gen_and_grade q0.5 --steps 4 --threshold 0.5 ;;
    q0.3) gen_and_grade q0.3 --steps 4 --threshold 0.3 ;;
    steps) gen_and_grade steps0.3-0.5-0.5 --steps 4 --threshold_steps 0.3,0.5,0.5 ;;
    k1) gen_and_grade k1 --steps 1 ;;
  esac
done
echo "=== CODEPRO $NAME done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
