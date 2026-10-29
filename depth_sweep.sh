#!/bin/bash
# Depth-headroom diagnostic: accuracy of Ouro-2.6B-Thinking at uniform recurrence
# depth K (total_ut_steps), including K=5,6 beyond the trained depth of 4.
#   TASK=gsm8k DATA=.../gsm8k_test.jsonl N=300 STEPS="4 2 1 3 6 5" BATCH=6 MAXNEW=2048 GPU=3 bash depth_sweep.sh
# Appends one RESULT json line per depth to $TMP/depth_${TASK}.txt and keeps generations in $TMP/depth_${TASK}_k${K}.jsonl
set -u
: "${TASK:?}"; : "${DATA:?}"; : "${GPU:?}"
N=${N:-300}; STEPS=${STEPS:-"4 2 1 3 6 5"}; BATCH=${BATCH:-6}; MAXNEW=${MAXNEW:-2048}
CKPT=${CKPT:-/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
REP=$TMP/depth_${TASK}.txt
cd "$WT" || exit 1
echo "=== DEPTH SWEEP $TASK n=$N steps=($STEPS) batch=$BATCH max_new=$MAXNEW gpu=$GPU ckpt=$(basename "$CKPT") $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
for K in $STEPS; do
  echo "--- K=$K start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" \
    --task "$TASK" --steps "$K" --n "$N" --batch "$BATCH" --max_new "$MAXNEW" \
    --out "$TMP/depth_${TASK}_k${K}.jsonl" > "$TMP/depth_${TASK}_k${K}.log" 2>&1
  grep '^RESULT' "$TMP/depth_${TASK}_k${K}.log" | tee -a "$REP"
  grep -q '^RESULT' "$TMP/depth_${TASK}_k${K}.log" || { echo "K=$K FAILED: $(grep -E 'Error|error' "$TMP/depth_${TASK}_k${K}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"; }
done
echo "=== DEPTH SWEEP $TASK done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
"$PY" - "$REP" <<'PYEOF'
import json, sys
rows = [json.loads(l[7:]) for l in open(sys.argv[1]) if l.startswith("RESULT ")]
print(f"{'K':>2} {'acc':>6} {'parse':>6} {'trunc':>6} {'tokens':>7} {'tok/s':>6}")
for r in sorted(rows, key=lambda r: r["steps"]):
    print(f"{r['steps']:>2} {r['acc']:6.3f} {r['parse_rate']:6.3f} {r['trunc_rate']:6.3f} {r['mean_tokens']:7.0f} {r['tok_s']:6.1f}")
PYEOF
