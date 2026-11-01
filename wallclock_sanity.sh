#!/bin/bash
# Sanity + follow-up for the true-early-exit collapse of the per-step rule (26% vs 67% gather):
#   GPU 2: list thresholds [1,1,1,1] must reproduce full depth; [0.3,0.3,0.3,1] must match scalar 0.3
#          (same 40 items as the wall-clock run) -> isolates "bug in the list path" from "KV approximation"
#   GPU 3: milder per-step rule (0.3,0.5,0.5) under true exit on items 0-99 (exits mostly at step >= 2)
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
REP=$TMP/wallclock_sanity.txt; : > "$REP"
cd "$WT" || exit 1
run() {  # run <gpu> <label> <n> <extra args...>
  local gpu=$1 label=$2 n=$3; shift 3
  echo "--- $label (n=$n, gpu $gpu) start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$TMP/cruxeval.jsonl" --task crux --steps 4 \
    --n "$n" --batch 1 --max_new 2048 "$@" --out "$TMP/wcs_${label}.jsonl" > "$TMP/wcs_${label}.log" 2>&1
  grep '^RESULT' "$TMP/wcs_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
}
(
  run 2 list_1111 40 --cfg "decode_early_exit_threshold=[1.0,1.0,1.0,1.0]"
  run 2 list_0.3x3 40 --cfg "decode_early_exit_threshold=[0.3,0.3,0.3,1.0]"
  run 2 scalar_0.3 40 --cfg decode_early_exit_threshold=0.3
) &
(
  run 3 steps0.3-0.5-0.5 100 --cfg "decode_early_exit_threshold=[0.3,0.5,0.5,1.0]"
) &
wait
echo "reference (items 0-99, true exit): full 73.0 @6.57 tok/s | gate0.3 68.0 @10.30 | steps0.1-0.5-0.5 26.0 @10.91 (45% truncated)" | tee -a "$REP"
echo "=== SANITY done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
