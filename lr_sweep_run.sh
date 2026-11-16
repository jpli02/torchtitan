#!/bin/bash
# Learning-rate debug: one terminal-SFT run from v3 on the take-turns r1 mix with the validation probe on
# (torchtitan/components/val_probe.py: held-out token loss before training and every VAL_EVERY steps).
# Everything is the recipe of train_tvary_r1_4ep.sh except what the variables change.
#   GPU=1 LR=5e-6 STEPS=1072 TAG=lr5e-6_const [DECAY=0.0] [WARMUP=50] [EXTRA_ARGS="--optimizer.beta2 0.999"] bash lr_sweep_run.sh
#   DECAY = lr_scheduler.decay_ratio: 0.0 keeps the LR constant after warm-up, 0.9 is the recipe's cosine decay.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
GPU=${GPU:-1}
LR=${LR:-2e-5}
STEPS=${STEPS:-1072}
DECAY=${DECAY:-0.0}
WARMUP=${WARMUP:-50}
TAG=${TAG:-lr${LR}_const}
INIT=${INIT:-/home/jli199/boptim_scratch/ouro_tb80sft26b_v3/checkpoint/step-20000}
SHUF=${DATA:-$TMP/tvary_r1_mixed_shuf1.jsonl}
DUMP=/home/jli199/boptim_scratch/ouro_lrsweep_${TAG}
LOG=/home/jli199/terminal_bench_eval/logs/ouro_lrsweep_${TAG}_train.log
[ -s "$SHUF" ] || { echo "ABORT no data at $SHUF"; exit 1; }
[ -d "$INIT" ] || { echo "ABORT no init at $INIT"; exit 1; }
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -n 1)
[ "${free:-0}" -ge "${MINFREE:-40000}" ] || { echo "ABORT gpu $GPU has ${free:-?} MiB free"; exit 1; }
cd "$WT" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU
export OURO_SFT_MIX=tb80sft OURO_SFT_LOCAL_JSONL=$SHUF
unset OURO_INIT_FROM OURO_INIT_DCP OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS OURO_SFT_PROSE_KEEP OURO_OPT_PROBE
export WANDB_MODE=offline WANDB_PROJECT=ouro-tb WANDB_RUN_NAME=lrsweep_$TAG WANDB_RUN_GROUP=lr-debug
export WANDB_DIR=/home/jli199/boptim_scratch/wandb
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OURO_VAL_FILE=${VAL_FILE:-$TMP/lrval_small_rendered.jsonl}
export OURO_VAL_EVERY=${VAL_EVERY:-134}
export OURO_VAL_OUT=$TMP/valprobe_${TAG}.jsonl
case "$DUMP" in /home/jli199/boptim_scratch/ouro_lrsweep_*) rm -rf "$DUMP" ;; *) echo "ABORT unexpected dump $DUMP"; exit 1 ;; esac
mkdir -p "$DUMP"; : > "$OURO_VAL_OUT"
echo "lr sweep $TAG: lr $LR, $STEPS steps, decay_ratio $DECAY, warmup $WARMUP, gpu $GPU, start $(date '+%m-%d %H:%M')"
"$PY" -m torch.distributed.run --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 -m torchtitan.train \
  --module ouro --config ouro_2_6b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --checkpoint.initial_load_path "$INIT" --checkpoint.initial_load_in_hf \
  --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps "$STEPS" --training.seq_len 4096 \
  --optimizer.lr "$LR" --lr_scheduler.warmup_steps "$WARMUP" --lr_scheduler.decay_ratio "$DECAY" ${EXTRA_ARGS:-} \
  --checkpoint.interval "${CKPT_INTERVAL:-1000000}" --metrics.log_freq 10 --metrics.enable_wandb \
  > "$LOG" 2>&1
echo "lr sweep $TAG exited $? $(date '+%m-%d %H:%M'); validation rows: $(wc -l < "$OURO_VAL_OUT")"
