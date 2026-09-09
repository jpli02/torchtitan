#!/bin/bash
# SFT from the pretrained base on the TB-80-targeted set (build_tb80_sft.py),
# then the 12-task eval ungated AND gated, sequentially on the same GPU.
#
# Recipe held to the traj20k one -- the best real result so far -- so the only
# variable is the data:
#   base       Ouro-1.4B-Thinking, pretrained
#   steps      20k (STEPS overridable), LR 2e-5, 50-step warmup, bs 1, seq 4096
#   ckpts      every 5k for a curve
#   data       tb80sft: verified multi-turn TB-2 trajectories (disjoint tasks,
#              re-rendered terminus-2, x5 upsample, cap 40/task) + TerminalTraj
#              finished trajectories reweighted to TB-80's category mix
#
# Comparison points on the 12 tasks:
#   continue10k 5/24 (45.5%)   traj20k 5/24 (54.3%)   gated traj20k 7/24 (58.6%)
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
DUMP=/home/jli199/boptim_scratch/ouro_tb80sft
CLEAN=/home/jli199/boptim_scratch/ouro_tb80sft_clean
JSONL=${JSONL:-/home/jli199/boptim_scratch/tb80sft/train.jsonl}
OUT=$TMP/tb80sft_chain.txt
STEPS=${STEPS:-20000}
: > "$OUT"
cd "$WT" || exit 1
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a
[ -s "$JSONL" ] || { echo "ABORT - no data at $JSONL" | tee -a "$OUT"; exit 1; }

gpu=""
for _ in $(seq 1 720); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, '$2+0>=31000 {print $1; exit}')
  [ -n "$gpu" ] && break
  sleep 60
done
[ -n "$gpu" ] || { echo "ABORT - no GPU with 31GB free after 12h" | tee -a "$OUT"; exit 1; }
echo "[tb80sft] training on gpu $gpu, $STEPS steps, data $JSONL $(date '+%m-%d %H:%M')" | tee -a "$OUT"

export CUDA_VISIBLE_DEVICES=$gpu
export OURO_SFT_MIX=tb80sft
export OURO_SFT_LOCAL_JSONL=$JSONL
unset OURO_INIT_FROM OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
export WANDB_MODE=online WANDB_PROJECT=ouro-terminal-sft
export WANDB_RUN_NAME="ouro-tb80sft-${STEPS}"
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
  --checkpoint.interval 5000 --metrics.log_freq 100 \
  > /home/jli199/terminal_bench_eval/logs/tb80sft_train.log 2>&1
echo "training exited $(date '+%m-%d %H:%M')" | tee -a "$OUT"

SRC=$(ls -d "$DUMP"/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$SRC" ] || { echo "ABORT - no checkpoint" | tee -a "$OUT"; exit 1; }
/home/jli199/torchtitan/.venv/bin/python scripts/push_ouro_sft_to_hub.py \
  --checkpoint "$SRC" --base_model /home/jli199/torchtitan/assets/hf/Ouro-1.4B-Thinking \
  --repo_id jaslee/tmp --dry_run >/dev/null 2>&1
rm -rf "$CLEAN"; mkdir -p "$CLEAN"
for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json \
         tokenizer_config.json special_tokens_map.json vocab.json merges.txt \
         model.safetensors.index.json model-00001-of-00001.safetensors; do
  [ -e "$SRC/$f" ] && ln -s "$SRC/$f" "$CLEAN/$f"
done
echo "staged $CLEAN from $SRC" | tee -a "$OUT"

GPU=$gpu GATE=0 CKPT="$CLEAN" NAME=tb80sft PORT=8102 ATT=2 MINFREE=28000 \
  bash eval_gated.sh >> "$OUT" 2>&1
GPU=$gpu GATE=1 CKPT="$CLEAN" NAME=gated_tb80sft PORT=8103 ATT=2 MINFREE=28000 \
  bash eval_gated.sh >> "$OUT" 2>&1
echo "--- chain done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
