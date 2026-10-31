#!/bin/bash
# Teacher-forced fit of the take-turns r1 4-true-epoch checkpoints on their own training rows
# (new rows vs v3 replay rows), to settle the underfitting question independently of the
# load-confounded pass@1 evals. Epoch 1 and 2 are already staged; epoch 4 is the final
# model-only save (stage it); epoch 3 is a DCP save (skipped).
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
DUMP=/home/jli199/boptim_scratch/ouro_tvary_r1_4ep
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
GPU=${GPU:-2}
REP=$TMP/fit_4ep.txt; : > "$REP"
cd "$WT" || exit 1
SRC=$DUMP/checkpoint/step-2144; CLEAN=${DUMP}_ep4_clean
if [ ! -f "$CLEAN/config.json" ]; then
  mkdir -p "$CLEAN" && cp "$SRC"/model*.safetensors "$CLEAN"/ 2>/dev/null
  [ -f "$SRC/model.safetensors.index.json" ] && cp "$SRC/model.safetensors.index.json" "$CLEAN"/
  for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$CLEAN/$f"; done
fi
ls "$CLEAN"/model*.safetensors >/dev/null 2>&1 || { echo "ABORT: epoch-4 weights not found under $SRC" | tee -a "$REP"; exit 1; }
echo "=== FIT of 4-true-epoch checkpoints on tvary_r1 rows (new vs replay) $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
for k in 1 2 4; do
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u fit_eval.py "${DUMP}_ep${k}_clean" "$TMP/tvary_r1_rendered.jsonl" "$TMP/fit_r1_4ep_ep${k}.json" > "$TMP/fit_r1_4ep_ep${k}.log" 2>&1
  echo "--- epoch $k:" | tee -a "$REP"
  grep -E '^  (new|replay) ' "$TMP/fit_r1_4ep_ep${k}.log" | tee -a "$REP"
done
echo "reference: v3 before continuation new 85.1 pct / replay 94.5 pct; r1 @1.67 epochs new 94.7 / replay 96.7" | tee -a "$REP"
echo "=== FIT 4ep done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
