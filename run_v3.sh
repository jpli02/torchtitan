#!/bin/bash
# Train Ouro-1.4B-Thinking on tb80sft v3 (the skill-thickened set), export an
# HF serving dir, then task-parallel eval on the 12 tasks across free GPUs.
#
# Same recipe as v2/traj20k so only the DATA differs (v3 = v2 + skill_mine
# rows for qemu/tmux/jupyter/fasttext). 20k steps, LR 2e-5, warmup 50, bs1,
# seq 4096, checkpoint every 5k.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
TAG=tb80sft_v3
JSONL=/home/jli199/boptim_scratch/tb80sft/train_v3.jsonl
STEPS=${STEPS:-20000}
DUMP=/home/jli199/boptim_scratch/ouro_${TAG}
CLEAN=/home/jli199/boptim_scratch/ouro_${TAG}_clean
OUT=$TMP/${TAG}_chain.txt
MINFREE_TRAIN=${MINFREE_TRAIN:-31000}
: > "$OUT"
cd "$WT" || exit 1
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a
[ -s "$JSONL" ] || { echo "ABORT - no data at $JSONL" | tee -a "$OUT"; exit 1; }

gpu=""
for _ in $(seq 1 1440); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, -v m="$MINFREE_TRAIN" '$2+0>=m {print $1; exit}')
  [ -n "$gpu" ] && break
  sleep 60
done
[ -n "$gpu" ] || { echo "ABORT - no GPU with ${MINFREE_TRAIN}MiB after 24h" | tee -a "$OUT"; exit 1; }
echo "[$TAG] training on gpu $gpu, $STEPS steps, data $JSONL ($(wc -l < "$JSONL") rows) $(date '+%m-%d %H:%M')" | tee -a "$OUT"

export CUDA_VISIBLE_DEVICES=$gpu
export OURO_SFT_MIX=tb80sft
export OURO_SFT_LOCAL_JSONL=$JSONL
unset OURO_INIT_FROM OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
# WANDB_MODE=offline captures/buffers the process console, so torchtitan's
# step logs never reach the redirected file (frozen log despite healthy
# training at ~3.3s/step). Disable wandb; rely on the log + tb events.
export WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p "$DUMP"

/home/jli199/torchtitan/.venv/bin/torchrun \
  --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 \
  -m torchtitan.train \
  --module ouro --config ouro_1_4b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps "$STEPS" \
  --checkpoint.interval 5000 --metrics.log_freq 20 \
  > /home/jli199/terminal_bench_eval/logs/${TAG}_train.log 2>&1
echo "training exited $(date '+%m-%d %H:%M')" | tee -a "$OUT"

SRC=$(ls -d "$DUMP"/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$SRC" ] || { echo "ABORT - no checkpoint" | tee -a "$OUT"; exit 1; }
# torchtitan's checkpoint saves ONLY the model weights (+ a sharded/ DCP copy);
# it has no config/modeling/tokenizer, so those must come from the base model
# assets, not from SRC. (The earlier symlink-from-SRC staging produced a
# weights-only dir and every eval server VOIDed with nothing to load.)
BASE=/home/jli199/torchtitan/assets/hf/Ouro-1.4B-Thinking
rm -rf "$CLEAN"; mkdir -p "$CLEAN"
cp "$SRC"/model-00001-of-00001.safetensors "$SRC"/model.safetensors.index.json "$CLEAN"/
for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json \
         tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do
  cp "$BASE/$f" "$CLEAN/$f"
done
echo "staged $CLEAN from $SRC + base aux ($(ls "$CLEAN" | wc -l) files)" | tee -a "$OUT"

# task-parallel eval on the 12 tasks, ungated then gated
CKPT="$CLEAN" BASENAME="$TAG" GATE=0 MINFREE=28000 bash eval_parallel.sh >> "$OUT" 2>&1
CKPT="$CLEAN" BASENAME="$TAG" GATE=1 MINFREE=28000 bash eval_parallel.sh >> "$OUT" 2>&1
echo "--- $TAG chain done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
