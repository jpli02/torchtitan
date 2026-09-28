#!/bin/bash
# TRAIN-ON-TEST DIAGNOSTIC -- contaminated by construction, never a score.
#
# Overfit the 12 eval tasks' own oracle solutions (build_oracle12.py), then run
# the SAME 12 tasks ungated. Answers one question: given perfect demonstrations
# of these exact tasks, can the model execute them through the agent loop?
#
# Init from continue10k, not the base: it already knows terminus-2 format, so
# any failure isolates to EXECUTION rather than to learning the output shape
# from 12 examples. 12 rows pack into ~6-10 windows per epoch, so 300 steps is
# ~30-50 epochs -- deliberate memorisation. LR 1e-5, continuation.
#
# Waits for a GPU with >=31GB free (training peaks at 29.1GB), then trains, then
# stages a servable copy and evals. One script, sequential, so the eval can
# never race a second copy of itself.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
DUMP=/home/jli199/boptim_scratch/ouro_oracle12
CLEAN=/home/jli199/boptim_scratch/ouro_oracle12_clean
OUT=$TMP/oracle12_chain.txt
STEPS=${STEPS:-300}
: > "$OUT"
cd "$WT" || exit 1
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a

gpu=""
for _ in $(seq 1 480); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, '$2+0>=31000 {print $1; exit}')
  [ -n "$gpu" ] && break
  sleep 60
done
[ -n "$gpu" ] || { echo "ABORT - no GPU with 31GB free after 8h" | tee -a "$OUT"; exit 1; }
echo "[oracle12] training on gpu $gpu, $STEPS steps $(date '+%m-%d %H:%M')" | tee -a "$OUT"

export CUDA_VISIBLE_DEVICES=$gpu
export OURO_SFT_MIX=oracle12
export OURO_SFT_LOCAL_JSONL=/home/jli199/boptim_scratch/oracle12/train.jsonl
export OURO_INIT_FROM=/home/jli199/boptim_scratch/ouro_c10k_clean
unset OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
export WANDB_MODE=online WANDB_PROJECT=ouro-terminal-sft
export WANDB_RUN_NAME="ouro-oracle12-TRAIN-ON-TEST"
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$DUMP"

/home/jli199/torchtitan/.venv/bin/torchrun \
  --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 \
  -m torchtitan.train \
  --module ouro --config ouro_1_4b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps "$STEPS" --optimizer.lr 1e-5 \
  --checkpoint.interval "$STEPS" --metrics.log_freq 10 \
  > /home/jli199/terminal_bench_eval/logs/oracle12_train.log 2>&1
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

# same GPU just freed by training; eval needs less than training did
GPU=$gpu GATE=0 CKPT="$CLEAN" NAME=oracle12 PORT=8099 ATT=2 MINFREE=28000 \
  bash eval_gated.sh >> "$OUT" 2>&1
echo "--- chain done $(date '+%m-%d %H:%M') --- LABEL: TRAIN-ON-TEST" | tee -a "$OUT"
