#!/bin/bash
# LiveCodeBench (ICLR 2025; v5+v6 release problems, 342 = 84 easy / 104 medium / 154 hard) router-headroom
# pipeline in DIRECT mode (forced ```python block, no reasoning preamble), one GPU; each point graded by
# executing the official tests in docker python:3.12-slim (lcb_grade.py).
#   NAME=lcb342 DATA=$TMP/lcb_all.jsonl N=342 GPU=1 bash lcb_pipeline.sh
# Points: K=4 (+oracle/features), K=2, gate q=0.5, gate q=0.3, per-step (0.3,0.5,0.5), K=1. Report: $TMP/bench_${NAME}.txt
set -u
: "${NAME:?}"; : "${DATA:?}"; : "${GPU:?}"
N=${N:-342}; MAXNEW=${MAXNEW:-1024}; BATCH=${BATCH:-6}
EXTRA=${EXTRA:-"--assistant_prefix \`\`\`python\\n"}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
S=/srv/scratch/jli199/lcb_grade_stage; mkdir -p "$S"; chmod 777 "$S"   # docker cannot mount the job dir; /tmp is small
REP=$TMP/bench_${NAME}.txt
cd "$WT" || exit 1
[ -f "$S/$(basename "$DATA")" ] || cp "$DATA" "$S"/
gen_and_grade() {  # gen_and_grade <label> <extra depth_eval args...>
  local label=$1; shift
  echo "--- $label start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task lcb \
    --n "$N" --batch "$BATCH" --max_new "$MAXNEW" $EXTRA "$@" --out "$TMP/bench_${NAME}_${label}.jsonl" > "$TMP/bench_${NAME}_${label}.log" 2>&1
  grep '^RESULT' "$TMP/bench_${NAME}_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
  cp "$WT/lcb_grade.py" "$TMP/bench_${NAME}_${label}.jsonl" "$S"/
  docker run --rm --network none --memory 12g --cpus 4 -v "$S":/w python:3.12-slim \
    python /w/lcb_grade.py "/w/$(basename "$DATA")" "/w/bench_${NAME}_${label}.jsonl" "/w/bench_${NAME}_${label}.graded.json" 2>&1 \
    | grep -E '^GRADED' | sed "s/^/$label /" | tee -a "$REP"
  cp "$S/bench_${NAME}_${label}.graded.json" "$TMP/" 2>/dev/null
}
echo "=== LCB $NAME direct (n=$N, max_new=$MAXNEW, batch=$BATCH, gpu $GPU) start $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
gen_and_grade k4 --steps 4
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" oracle_depth.py "$CKPT" "$TMP/bench_${NAME}_k4.jsonl" --task lcb --data "$DATA" \
  --out "$TMP/bench_${NAME}_oracle.json" --dump "$TMP/feat_${NAME}_k4.pt" > "$TMP/bench_${NAME}_oracle.log" 2>&1
grep -vE 'Warning|warn|deprecated|^ORACLE|gens,|dumped' "$TMP/bench_${NAME}_oracle.log" | tail -9 | tee -a "$REP"
gen_and_grade k2 --steps 2
gen_and_grade q0.5 --steps 4 --threshold 0.5
gen_and_grade q0.3 --steps 4 --threshold 0.3
gen_and_grade steps0.3-0.5-0.5 --steps 4 --threshold_steps 0.3,0.5,0.5
gen_and_grade k1 --steps 1
echo "=== LCB $NAME done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
