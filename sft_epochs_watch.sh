#!/bin/bash
# Watcher half of sft_epochs_eval.sh, for a training already running: waits for each
# epoch checkpoint (k-th "Finished saving" line in the titan log), stages it as an HF
# dir (DCP -> safetensors via export_dcp_to_hf.py), evals it on the 12 tasks, summarises.
#   TAG=ds_flash_ep SPE=685 EPOCHS=4 TRAIN_PID=<pid> TITANLOG=<torchelastic stdout.log> bash sft_epochs_watch.sh
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
TAG=${TAG:?}; SPE=${SPE:?}; EPOCHS=${EPOCHS:-4}; TRAIN_PID=${TRAIN_PID:?}; TITANLOG=${TITANLOG:?}
DUMP=/home/jli199/boptim_scratch/ouro_${TAG}; OUT=$TMP/${TAG}_chain.txt
cd "$WT" || exit 1
echo "[$TAG] watcher attached to pid $TRAIN_PID, log $TITANLOG $(date '+%m-%d %H:%M')" | tee -a "$OUT"
stage() {
  local src=$1 clean=$2
  rm -rf "$clean"; mkdir -p "$clean"
  if ls "$src"/model*.safetensors >/dev/null 2>&1; then
    cp "$src"/model*.safetensors "$clean"/
    [ -f "$src/model.safetensors.index.json" ] && cp "$src/model.safetensors.index.json" "$clean"/
  else
    local cg
    cg=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' \
         | awk -F, '$2+0>=20000 {print $2+0, $1}' | sort -rn | head -1 | awk '{print $2}')
    CUDA_VISIBLE_DEVICES="${cg:-0}" PYTHONPATH="$WT" "$PY" scripts/export_dcp_to_hf.py \
      --module ouro --config ouro_2_6b_thinking_terminal_sft \
      --checkpoint "$src" --hf_template "$BASE" --out "$clean" --dtype bf16 \
      > "$TMP/${TAG}_export_$(basename "$src").log" 2>&1
    rm -rf "$clean/sharded"
  fi
  for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json \
           tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$clean/$f"; done
  ls "$clean"/model*.safetensors >/dev/null 2>&1 || { echo "ABORT - staging $clean produced no weights" | tee -a "$OUT"; exit 1; }
}
for k in $(seq 1 "$EPOCHS"); do
  step=$((SPE * k)); src="$DUMP/checkpoint/step-$step"
  while :; do
    saved=$(grep -cE "Finished saving the checkpoint|Saving a model only checkpoint" "$TITANLOG" 2>/dev/null); saved=${saved:-0}
    { [ "$saved" -ge "$k" ] && [ -d "$src" ]; } && break
    ls "$src"/model*.safetensors >/dev/null 2>&1 && break
    kill -0 "$TRAIN_PID" 2>/dev/null || { echo "ABORT - training died before epoch $k: $(grep -m1 -E 'Error|OutOfMemory' "$TITANLOG" | cut -c1-140)" | tee -a "$OUT"; exit 1; }
    sleep 60
  done
  sleep 30
  loss=$(grep -oE "step: *$step .*loss: *[0-9.]+" "$TITANLOG" | tail -1 | sed 's/\x1b\[[0-9;]*m//g' | grep -oE "loss: *[0-9.]+")
  clean="${DUMP}_ep${k}_clean"; stage "$src" "$clean"
  echo "[$TAG] epoch $k (step $step, ${loss:-loss n/a}) staged $clean; eval starting $(date '+%m-%d %H:%M')" | tee -a "$OUT"
  CKPT="$clean" BASENAME="${TAG}_ep${k}" GATE=0 MINFREE=28000 bash eval_parallel.sh >> "$OUT" 2>&1
  echo "[$TAG] epoch $k eval done $(date '+%m-%d %H:%M')" | tee -a "$OUT"
done
echo "=== $TAG summary (12 tasks x 2 attempts, stock terminus-2) ===" | tee -a "$OUT"
for k in $(seq 1 "$EPOCHS"); do grep -E "^${TAG}_ep${k}: " "$OUT" | tail -1; done | tee -a "$OUT"
echo "--- $TAG done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
