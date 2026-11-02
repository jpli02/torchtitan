#!/bin/bash
# BigCodeBench (direct mode, forced code block) under TRUE decode-time early exit, batch 1:
# full depth | gate q=0.5 | gate q=0.3 on the first N tasks of the full-set sample; each graded.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
DATA=$TMP/bcb_full300.jsonl
GPU=${GPU:-3}; N=${N:-150}
S=/tmp/bcb_grade_stage; mkdir -p "$S"; chmod 777 "$S"
REP=$TMP/wallclock_bcb.txt; : > "$REP"
cd "$WT" || exit 1
run() {  # run <label> <extra args...>
  local label=$1; shift
  echo "--- $label start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task bcb --steps 4 \
    --n "$N" --batch 1 --max_new 768 --assistant_prefix '```python\n' "$@" --out "$TMP/wcb_${label}.jsonl" > "$TMP/wcb_${label}.log" 2>&1
  grep '^RESULT' "$TMP/wcb_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
  cp "$WT/bcb_grade.py" "$DATA" "$TMP/wcb_${label}.jsonl" "$S"/
  docker run --rm --network none --memory 6g --cpus 4 --user "$(id -u):$(id -g)" -v "$S":/w --entrypoint python \
    bigcodebench/bigcodebench-evaluate:latest /w/bcb_grade.py /w/bcb_full300.jsonl "/w/wcb_${label}.jsonl" "/w/wcb_${label}.graded.json" 2>&1 \
    | grep -E '^GRADED' | sed "s/^/$label /" | tee -a "$REP"
  cp "$S/wcb_${label}.graded.json" "$TMP/" 2>/dev/null
}
echo "=== BCB TRUE-EXIT wall-clock (direct mode, batch 1, n=$N, gpu $GPU) $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
run full
run gate0.5 --cfg decode_early_exit_threshold=0.5
run gate0.3 --cfg decode_early_exit_threshold=0.3
echo "=== BCB TRUE-EXIT done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
