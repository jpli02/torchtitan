#!/bin/bash
# Verify the deepseek-generated terminal dataset: continue-SFT Ouro-2.6B from the
# tb80sft-v3 checkpoint on the generated trajectories, then eval on the 12-task
# Terminal-Bench subset (gated + ungated). Compare to the v3 baseline (6/24
# ungated, 8/24 gated) to see whether the generated data moves pass@1.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
TAG=${TAG:-ds_verify}
JSONL=${JSONL:-/home/jli199/boptim_scratch/gen_out/deepseek_generated.jsonl}
INIT=${INIT:-}   # empty = start from the Ouro-2.6B-Thinking HF checkpoint (config default)
STEPS=${STEPS:-1000}
SEQLEN=${SEQLEN:-4096}
MINFREE=${MINFREE:-38000}
DUMP=/home/jli199/boptim_scratch/ouro_${TAG}
CLEAN=${DUMP}_clean
OUT=$TMP/${TAG}_chain.txt
: > "$OUT"
cd "$WT" || exit 1
[ -s "$JSONL" ] || { echo "ABORT - no data at $JSONL" | tee -a "$OUT"; exit 1; }
[ -n "$INIT" ] && { [ -d "$INIT" ] || { echo "ABORT - no init ckpt $INIT" | tee -a "$OUT"; exit 1; }; }
SRC_DESC=$([ -n "$INIT" ] && basename "$INIT" || echo "Ouro-2.6B-Thinking HF base")
echo "[$TAG] SFT from $SRC_DESC on $(wc -l < "$JSONL") rows, $STEPS steps $(date '+%m-%d %H:%M')" | tee -a "$OUT"

gpu="${GPU:-}"   # pin via GPU env; else auto-pick the card with most free memory
[ -z "$gpu" ] && for _ in $(seq 1 1440); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, -v m="$MINFREE" '$2+0>=m {print $2+0, $1}' | sort -rn | head -1 | awk '{print $2}')
  [ -n "$gpu" ] && break
  sleep 60
done
[ -n "$gpu" ] || { echo "ABORT - no GPU with ${MINFREE}MiB" | tee -a "$OUT"; exit 1; }
echo "[$TAG] gpu $gpu $(date '+%m-%d %H:%M')" | tee -a "$OUT"

export CUDA_VISIBLE_DEVICES=$gpu
export OURO_SFT_MIX=tb80sft
export OURO_SFT_LOCAL_JSONL=$JSONL
if [ -n "$INIT" ]; then export OURO_INIT_FROM=$INIT; else unset OURO_INIT_FROM; fi
unset OURO_INIT_DCP OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
export WANDB_MODE=disabled PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$DUMP"

/home/jli199/torchtitan/.venv/bin/torchrun \
  --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 -m torchtitan.train \
  --module ouro --config ouro_2_6b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps "$STEPS" --training.seq_len "$SEQLEN" \
  --checkpoint.interval "$STEPS" --metrics.log_freq 50 \
  > /home/jli199/terminal_bench_eval/logs/${TAG}_train.log 2>&1
echo "training exited $(date '+%m-%d %H:%M')" | tee -a "$OUT"

SRC=$(ls -d "$DUMP"/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$SRC" ] || { echo "ABORT - no checkpoint" | tee -a "$OUT"; exit 1; }
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
rm -rf "$CLEAN"; mkdir -p "$CLEAN"
cp "$SRC"/model*.safetensors "$CLEAN"/ 2>/dev/null
[ -f "$SRC/model.safetensors.index.json" ] && cp "$SRC/model.safetensors.index.json" "$CLEAN"/
for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json \
         tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do
  cp "$BASE/$f" "$CLEAN/$f"
done
echo "staged $CLEAN ($(ls "$CLEAN" | wc -l) files)" | tee -a "$OUT"

CKPT="$CLEAN" BASENAME="$TAG" GATE=0 MINFREE=28000 bash eval_parallel.sh >> "$OUT" 2>&1
CKPT="$CLEAN" BASENAME="$TAG" GATE=1 MINFREE=28000 bash eval_parallel.sh >> "$OUT" 2>&1
echo "--- $TAG done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
