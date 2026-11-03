#!/bin/bash
# GSM-Symbolic (ICLR 2025; main subset by default) under TRUE decode-time early exit, batch 1, first N items,
# 2,048-token budget, accuracy scored by depth_eval itself (boxed answer). Long chain-of-thought regime
# (about 1,000 tokens per answer), the third regime next to BigCodeBench (short code) and LiveCodeBench (medium).
#   GPU=4 POINTS="full" bash wallclock_gsm.sh ; GPU=5 POINTS="gate0.5" bash wallclock_gsm.sh   -> $TMP/wallclock_gsm.txt
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
DATA=${DATA:-$TMP/gsm_symbolic_main.jsonl}; TAG=${TAG:-gsmmain}
GPU=${GPU:-4}; N=${N:-100}; MAXNEW=${MAXNEW:-2048}
OFFSET=${OFFSET:-0}; SUF=""; [ "$OFFSET" != 0 ] && SUF="_o${OFFSET}"
POINTS=${POINTS:-"full gate0.5 gate0.3 gate0.7"}
REP=$TMP/wallclock_gsm.txt
cd "$WT" || exit 1
run() {  # run <point> <extra args...>
  local label=${TAG}_${1}${SUF}; shift
  echo "--- $label start $(date '+%m-%d %H:%M') (gpu $GPU, n $N, offset $OFFSET)" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task gsm8k --steps 4 \
    --n "$N" --batch 1 --max_new "$MAXNEW" --seed_offset "$OFFSET" "$@" --out "$TMP/wcg_${label}.jsonl" > "$TMP/wcg_${label}.log" 2>&1
  grep '^RESULT' "$TMP/wcg_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
  grep -q '^RESULT' "$TMP/wcg_${label}.log" || echo "$label FAILED: $(grep -E 'Error' "$TMP/wcg_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"
}
for p in $POINTS; do
  case "$p" in
    full) run full ;;
    gate0.7) run gate0.7 --cfg decode_early_exit_threshold=0.7 ;;
    gate0.5) run gate0.5 --cfg decode_early_exit_threshold=0.5 ;;
    gate0.3) run gate0.3 --cfg decode_early_exit_threshold=0.3 ;;
    gate0.2) run gate0.2 --cfg decode_early_exit_threshold=0.2 ;;
    steps0.3-0.5-0.5) run steps0.3-0.5-0.5 --cfg "decode_early_exit_threshold=[0.3,0.5,0.5,1.0]" ;;
    *) echo "unknown point $p" | tee -a "$REP" ;;
  esac
done
echo "=== GSM true-exit points [$POINTS] ($TAG, gpu $GPU) done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
