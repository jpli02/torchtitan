#!/bin/bash
# True-exit gate curve on the same 200 items (100-299) as the mild per-step rule and full depth:
# q=0.5 (GPU 2) and q=0.7 (GPU 3), batch 1, 2,048-token budget.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
REP=$TMP/wallclock_pareto.txt
cd "$WT" || exit 1
run() {
  local gpu=$1 label=$2 n=$3 off=$4; shift 4
  echo "--- $label (n=$n, offset=$off, gpu $gpu) start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$TMP/cruxeval.jsonl" --task crux --steps 4 \
    --n "$n" --seed_offset "$off" --batch 1 --max_new 2048 "$@" --out "$TMP/wcp_${label}.jsonl" > "$TMP/wcp_${label}.log" 2>&1
  grep '^RESULT' "$TMP/wcp_${label}.log" | sed "s/^RESULT /RESULT $label offset=$off /" | tee -a "$REP"
}
( run 2 gate0.5_o100 200 100 --cfg decode_early_exit_threshold=0.5 ) &
( run 3 gate0.7_o100 200 100 --cfg decode_early_exit_threshold=0.7 ) &
wait
echo "=== GATE o100 done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
