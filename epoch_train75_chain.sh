#!/bin/bash
# Pass rate of continuation epochs on ALL 75 training tasks (the ep4 / r1 protocol: one attempt, 1,800 s, 2,048-token turn cap,
# 12k context cap, lever-1 serving, ungated), one epoch after another on one card; tasks a harness cap leaves unfinished are re-run.
#   GPU=1 EPOCHS_LIST="6 5" bash epoch_train75_chain.sh      -> /srv/scratch/jli199/tb_runs/train75_ep{k}_1800, report train_tvary_r1_8ep.txt
set -u
: "${GPU:?}"; : "${EPOCHS_LIST:?}"
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
TAG=ouro_tvary_r1_8ep; DUMP=/home/jli199/boptim_scratch/$TAG; SPE=536
POOL=/home/jli199/boptim_scratch/tb_variants/pool; OUT=/srv/scratch/jli199/tb_runs
REP=$TMP/train_tvary_r1_8ep.txt
TASKS75=$(cat "$TMP/train_tasks_r1.txt")
export OURO_SEQ_CAP=12000
cd "$WT" || exit 1
stage() {  # stage <step-dir> <clean-dir>: interval saves are DCP -> export; a final save is safetensors
  local src=$1 clean=$2
  rm -rf "$clean"; mkdir -p "$clean"
  if ls "$src"/model*.safetensors >/dev/null 2>&1; then
    cp "$src"/model*.safetensors "$clean"/
    [ -f "$src/model.safetensors.index.json" ] && cp "$src/model.safetensors.index.json" "$clean"/
  else
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$WT" "$PY" scripts/export_dcp_to_hf.py \
      --module ouro --config ouro_2_6b_thinking_terminal_sft \
      --checkpoint "$src" --hf_template "$BASE" --out "$clean" --dtype bf16 > "$TMP/${TAG}_export_$(basename "$src").log" 2>&1
    rm -rf "$clean/sharded"
  fi
  for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$clean/$f"; done
  ls "$clean"/model*.safetensors >/dev/null 2>&1
}
for k in $EPOCHS_LIST; do
  step=$(( SPE * (k - 4) )); src="$DUMP/checkpoint/step-$step"; clean="${DUMP}_ep${k}_clean"
  if ! ls "$clean"/model*.safetensors >/dev/null 2>&1; then
    [ -d "$src" ] || { echo "epoch $k: no checkpoint at $src" | tee -a "$REP"; continue; }
    if ! stage "$src" "$clean"; then echo "epoch $k: staging FAILED" | tee -a "$REP"; continue; fi
    echo "epoch $k (step $step) staged $clean $(date '+%m-%d %H:%M')" | tee -a "$REP"
  fi
  NAME=train75_ep${k}_1800
  echo "=== epoch $k on all 75 training tasks x1, 1800 s, GPU $GPU start $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
  GPU=$GPU GATE=0 MAXTOK=2048 TBTIMEOUT=100000 CKPT="$clean" NAME=$NAME PORT=$((8180 + k)) ATT=1 MINFREE=${MINFREE:-38000} \
    DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=1800 TASKS="$TASKS75" bash eval_gated.sh > "$TMP/$NAME.log" 2>&1
  # tasks the harness cap left unfinished
  left=$(for t in $TASKS75; do [ "$(find "$OUT/$NAME/$t" -name results.json 2>/dev/null | wc -l)" -ge 1 ] || echo "$t"; done | tr '\n' ' ')
  if [ -n "$left" ]; then
    echo "epoch $k: $(echo $left | wc -w) tasks left after the cap; resuming $(date '+%m-%d %H:%M')" | tee -a "$REP"
    GPU=$GPU GATE=0 MAXTOK=2048 TBTIMEOUT=100000 CKPT="$clean" NAME=$NAME PORT=$((8180 + k)) ATT=1 MINFREE=${MINFREE:-38000} \
      DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=1800 TASKS="$left" bash eval_gated.sh > "$TMP/${NAME}_resume.log" 2>&1
  fi
  echo "epoch $k train75: $(grep -c 'OOM at step' /home/jli199/terminal_bench_eval/logs/${NAME}_server.log 2>/dev/null) out-of-memory turns of $(grep -c 'TURN prompt' /home/jli199/terminal_bench_eval/logs/${NAME}_server.log 2>/dev/null)" | tee -a "$REP"
  python3 -I tb_progress.py "$OUT" $NAME | tee -a "$REP"
  echo "epoch $k train75 done $(date '+%m-%d %H:%M')" | tee -a "$REP"
done
echo "[gpu $GPU] train75 chain for epochs [$EPOCHS_LIST] finished $(date '+%m-%d %H:%M')" | tee -a "$REP"
