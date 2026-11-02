#!/bin/bash
# BigCodeBench router-headroom pipeline (one GPU): generation points in gather mode, each graded
# in the official image, plus the oracle/features from the K=4 generations.
#   NAME=bcb300 DATA=$TMP/bcb_full300.jsonl N=300 GPU=2 MAXNEW=3072 BATCH=4 bash bcb_pipeline.sh
# Points: K=4, K=2, gate q=0.5, gate q=0.3, per-step (0.3,0.5,0.5). Report: $TMP/bench_${NAME}.txt
set -u
: "${NAME:?}"; : "${DATA:?}"; : "${GPU:?}"
N=${N:-300}; MAXNEW=${MAXNEW:-768}; BATCH=${BATCH:-8}
# Direct mode: the thinking-style model ignores "answer directly" instructions, but starting the
# assistant turn inside a code block skips the reasoning (258 tokens mean vs ~3,000). EXTRA="" for the
# thinking mode (needs >= 6k tokens).
EXTRA=${EXTRA:-"--assistant_prefix \`\`\`python\\n"}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
S=/tmp/bcb_grade_stage; mkdir -p "$S"; chmod 777 "$S"
REP=$TMP/bench_${NAME}.txt
cd "$WT" || exit 1
gen_and_grade() {  # gen_and_grade <label> <extra depth_eval args...>
  local label=$1; shift
  echo "--- $label start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task bcb \
    --n "$N" --batch "$BATCH" --max_new "$MAXNEW" $EXTRA "$@" --out "$TMP/bench_${NAME}_${label}.jsonl" > "$TMP/bench_${NAME}_${label}.log" 2>&1
  grep '^RESULT' "$TMP/bench_${NAME}_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
  cp "$WT/bcb_grade.py" "$DATA" "$TMP/bench_${NAME}_${label}.jsonl" "$S"/
  docker run --rm --network none --memory 6g --cpus 4 --user "$(id -u):$(id -g)" -v "$S":/w --entrypoint python \
    bigcodebench/bigcodebench-evaluate:latest /w/bcb_grade.py "/w/$(basename "$DATA")" "/w/bench_${NAME}_${label}.jsonl" "/w/bench_${NAME}_${label}.graded.json" 2>&1 \
    | grep -E '^GRADED' | sed "s/^/$label /" | tee -a "$REP"
  cp "$S/bench_${NAME}_${label}.graded.json" "$TMP/" 2>/dev/null
}
echo "=== BCB $NAME (n=$N, max_new=$MAXNEW, batch=$BATCH, gpu $GPU) start $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
gen_and_grade k4 --steps 4
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" oracle_depth.py "$CKPT" "$TMP/bench_${NAME}_k4.jsonl" --task bcb --data "$DATA" \
  --out "$TMP/bench_${NAME}_oracle.json" --dump "$TMP/feat_${NAME}_k4.pt" > "$TMP/bench_${NAME}_oracle.log" 2>&1
grep -vE 'Warning|warn|deprecated|^ORACLE|gens,|dumped' "$TMP/bench_${NAME}_oracle.log" | tail -9 | tee -a "$REP"
gen_and_grade k2 --steps 2
gen_and_grade q0.5 --steps 4 --threshold 0.5
gen_and_grade q0.3 --steps 4 --threshold 0.3
gen_and_grade steps0.3-0.5-0.5 --steps 4 --threshold_steps 0.3,0.5,0.5
echo "=== BCB $NAME done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
