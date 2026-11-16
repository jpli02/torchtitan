#!/bin/bash
# Learning-rate debug: a short run of the REAL terminal-SFT trainer with the optimizer probe on (torchtitan/components/
# opt_probe.py): for sampled parameter tensors, the update the optimizer really applied against the update exact fp32 AdamW
# would apply for the same gradients. The LR curve is the real run's (schedule computed for 2144 steps), only cut short.
#   GPU=1 INIT=<hf dir> STEPS=150 TAG=probe_ep4 [OPT=AdamW8bit] bash lr_probe_run.sh
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
GPU=${GPU:-1}
INIT=${INIT:-/home/jli199/boptim_scratch/ouro_tvary_r1_4ep_ep4_clean}
STEPS=${STEPS:-150}
TAG=${TAG:-probe_ep4}
OPT=${OPT:-AdamW8bit}
SHUF=$TMP/tvary_r1_mixed_shuf1.jsonl
DUMP=/home/jli199/boptim_scratch/ouro_lrdebug_${TAG}
LOG=$TMP/lrdebug_${TAG}_train.log
[ -s "$SHUF" ] || { echo "ABORT no data at $SHUF"; exit 1; }
[ -d "$INIT" ] || { echo "ABORT no init at $INIT"; exit 1; }
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -n 1)
[ "${free:-0}" -ge "${MINFREE:-38000}" ] || { echo "ABORT gpu $GPU has ${free:-?} MiB free"; exit 1; }
cd "$WT" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU
export OURO_SFT_MIX=tb80sft OURO_SFT_LOCAL_JSONL=$SHUF
unset OURO_INIT_FROM OURO_INIT_DCP OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS OURO_SFT_PROSE_KEEP
export WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OURO_OPT_PROBE=$TMP/opt_probe_${TAG}.jsonl
rm -rf "$DUMP"; mkdir -p "$DUMP"
echo "probe run $TAG: init $INIT, $STEPS steps, optimizer $OPT, gpu $GPU, start $(date '+%m-%d %H:%M')"
"$PY" -m torch.distributed.run --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 -m torchtitan.train \
  --module ouro --config ouro_2_6b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --checkpoint.initial_load_path "$INIT" --checkpoint.initial_load_in_hf \
  --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps "$STEPS" --training.seq_len 4096 --lr_scheduler.total_steps 2144 \
  --optimizer.name "$OPT" ${EXTRA_ARGS:-} \
  --checkpoint.interval 1000000 --metrics.log_freq 10 --metrics.enable_wandb \
  > "$LOG" 2>&1
echo "probe run $TAG exited $? $(date '+%m-%d %H:%M'); probe rows: $(wc -l < "$OURO_OPT_PROBE" 2>/dev/null)"
rm -rf "$DUMP"
