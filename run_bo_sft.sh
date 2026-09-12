#!/bin/bash
# SFT Qwen3-1.7B into a boptim-agent BO optimizer (config qwen3_1_7b_bo_sft).
#
# Data: single-turn rows from bo_replay_format.py (optimizer prompt -> JSON
# reply). Read through the terminal_agent_sft dataset's local-JSONL route;
# OURO_SFT_MIX=oracle12 is just the "100% local JSONL" mix (the name is
# historical, the loader is model-agnostic: it applies the Qwen chat template
# and masks everything but assistant tokens).
#
#   TAG=bo_smoke STEPS=5 GPU=9 JSONL=/home/jli199/boptim_scratch/bo_sft/cifar.jsonl bash run_bo_sft.sh
#   TAG=bo_v1    STEPS=1000 GPU=9 JSONL=/home/jli199/boptim_scratch/bo_sft/train.jsonl bash run_bo_sft.sh
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
TAG=${TAG:-bo_smoke}
GPU=${GPU:-9}
STEPS=${STEPS:-5}
JSONL=${JSONL:-/home/jli199/boptim_scratch/bo_sft/cifar.jsonl}
DUMP=/home/jli199/boptim_scratch/qwen_${TAG}
LOG=/home/jli199/terminal_bench_eval/logs/qwen_${TAG}_train.log
OUT=$TMP/qwen_${TAG}.txt
CKPT_EVERY=${CKPT_EVERY:-500}
: > "$OUT"
cd "$WT" || exit 1
[ -s "$JSONL" ] || { echo "ABORT - no data at $JSONL" | tee -a "$OUT"; exit 1; }
echo "[qwen-bo] $TAG on gpu $GPU, $STEPS steps, data $JSONL ($(wc -l < "$JSONL") rows) $(date '+%m-%d %H:%M')" | tee -a "$OUT"

export CUDA_VISIBLE_DEVICES=$GPU
export OURO_SFT_MIX=oracle12
export OURO_SFT_LOCAL_JSONL=$JSONL
unset OURO_INIT_FROM OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
export WANDB_MODE=${WANDB_MODE:-online} WANDB_PROJECT=qwen-bo-sft
export WANDB_RUN_NAME="qwen1.7b-${TAG}-${STEPS}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p "$DUMP"

/home/jli199/torchtitan/.venv/bin/torchrun \
  --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 \
  -m torchtitan.train \
  --module qwen3 --config qwen3_1_7b_bo_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps "$STEPS" \
  --checkpoint.interval "$CKPT_EVERY" --metrics.log_freq 1 \
  > "$LOG" 2>&1
rc=$?
echo "training exited rc=$rc $(date '+%m-%d %H:%M')" | tee -a "$OUT"
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -oE "step: +[0-9]+ +loss: +[0-9.]+" | sed -n '1p;$p' | tee -a "$OUT"
ls "$DUMP/checkpoint" 2>/dev/null | tee -a "$OUT"

# Serving dir: the HF export has only the safetensors (+ a sharded/ DCP copy
# that makes the HF loader spin); add config/tokenizer from the base model.
SRC=$(ls -d "$DUMP"/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
CLEAN=${DUMP}_clean
if [ -n "$SRC" ] && [ -f "$SRC/model.safetensors.index.json" ]; then
  rm -rf "$CLEAN"; mkdir -p "$CLEAN"
  cp "$SRC"/model*.safetensors "$SRC"/model.safetensors.index.json "$CLEAN"/
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    cp "$WT/assets/hf/Qwen3-1.7B/$f" "$CLEAN"/ 2>/dev/null
  done
  /home/jli199/torchtitan/.venv/bin/python "$WT/fix_tied_export.py" "$CLEAN" | tee -a "$OUT"
  echo "serving dir: $CLEAN ($(du -sh "$CLEAN" | cut -f1))" | tee -a "$OUT"
fi
echo "done" | tee -a "$OUT"
