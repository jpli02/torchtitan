#!/bin/bash
# Continue the ep4 checkpoint (take-turns r1 data, 4 true epochs) for EPOCHS more true epochs on the SAME shuffled 1:1 mixture
# (tvary_r1_mixed_shuf1.jsonl: 112 r1 rows + 112 v3 replay rows, 2,194,525 tokens -> 536 packed steps per epoch), weights-only
# load (fresh AdamW8bit state and schedule, as every continuation here), checkpoint at every epoch boundary. The eval chains
# (epoch_eval_chain.sh) stage each checkpoint and evaluate it as it lands.
#   GPU=5 EPOCHS=4 bash train_ep4_more.sh     -> ouro_tvary_r1_8ep/checkpoint/step-{536,1072,1608,2144} = epochs 5..8
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
INIT=/home/jli199/boptim_scratch/ouro_tvary_r1_4ep_ep4_clean
TAG=ouro_tvary_r1_8ep
DUMP=/home/jli199/boptim_scratch/${TAG}
SHUF=$TMP/tvary_r1_mixed_shuf1.jsonl
REP=$TMP/train_tvary_r1_8ep.txt
GPU=${GPU:-5}
SPE=536; EPOCHS=${EPOCHS:-4}; STEPS=$((SPE*EPOCHS))
: > "$REP"
echo "=== TRAIN ep4 -> +$EPOCHS epochs ($TAG) start $(date '+%m-%d %H:%M') gpu=$GPU steps=$STEPS (spe=$SPE) init=$INIT ===" | tee -a "$REP"
free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | awk -F, -v g="$GPU" '$1==g {print $2}')
[ "${free:-0}" -ge 40000 ] || { echo "ABORT gpu $GPU has ${free:-?}MiB free" | tee -a "$REP"; exit 1; }
[ -s "$SHUF" ] || { echo "ABORT no data at $SHUF" | tee -a "$REP"; exit 1; }
cd "$WT" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU
export OURO_SFT_MIX=tb80sft OURO_SFT_LOCAL_JSONL=$SHUF
unset OURO_INIT_FROM OURO_INIT_DCP OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS OURO_SFT_PROSE_KEEP
export WANDB_MODE=offline WANDB_PROJECT=ouro-tb WANDB_RUN_NAME=$TAG WANDB_RUN_GROUP=take-turns-8ep
export WANDB_DIR=/home/jli199/boptim_scratch/wandb
export PYTORCH_ALLOC_CONF=expandable_segments:True
rm -rf "$DUMP"; mkdir -p "$DUMP"
echo "training on gpu $GPU $(date '+%m-%d %H:%M')" | tee -a "$REP"
"$PY" -m torch.distributed.run --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 -m torchtitan.train \
  --module ouro --config ouro_2_6b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --checkpoint.initial_load_path "$INIT" --checkpoint.initial_load_in_hf \
  --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps "$STEPS" --training.seq_len 4096 \
  --checkpoint.interval "$SPE" --metrics.log_freq 10 --metrics.enable_wandb \
  > /home/jli199/terminal_bench_eval/logs/${TAG}_train.log 2>&1
echo "training exited $(date '+%m-%d %H:%M'); checkpoints: $(ls -d "$DUMP"/checkpoint/step-* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')" | tee -a "$REP"
touch "$DUMP/TRAIN_DONE"
