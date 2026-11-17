#!/bin/bash
# ONE training run from the BASE model (user rule 2026-11-16: every training starts from base Ouro-2.6B-Thinking):
#   data   tb80sft/train_frombase_v3r1_shuf7.jsonl  (build_frombase_mix.py: v3's 7,682 rows minus 60 held out, plus the 112
#          take-turns r1 trajectories, shuffled; 90.87M tokens)
#   steps  22184 = exactly one pass of 4096-token windows (v3 itself read 0.905 of an UNSHUFFLED pass and so never saw the
#          358 skill rows at the end of its file)
#   lr     the recipe's 2e-5 with 50 warm-up steps and cosine decay to 2e-6 (the 2e-5 problem found on 11-16 is specific
#          to repeated passes over 224 rows from an already fitted model; from base every row is seen once)
#   held-out validation inside training every VAL_EVERY steps (val_probe), DCP checkpoints every quarter pass.
# Then: stage the final checkpoint and run the Terminal-Bench TEST evaluation (11 tasks x 2), run directory on /srv/scratch.
#   GPU=1 bash train_frombase.sh            (about 41 hours of training on a whole card, then about 6 hours of evaluation)
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
S=/home/jli199/boptim_scratch
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
GPU=${GPU:-1}
LR=${LR:-2e-5}
STEPS=${STEPS:-22184}
TAG=${TAG:-frombase_v3r1_p1}
DATA=${DATA:-$S/tb80sft/train_frombase_v3r1_shuf7.jsonl}
VAL=${VAL:-$S/lr_debug/lrval_frombase_rendered.jsonl}
DUMP=$S/ouro_${TAG}
LOG=/home/jli199/terminal_bench_eval/logs/ouro_${TAG}_train.log
REP=$TMP/train_${TAG}.txt
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
[ -s "$DATA" ] && [ -s "$VAL" ] && [ -d "$BASE" ] || { echo "ABORT missing data, validation file or base model"; exit 1; }
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -n 1)
[ "${free:-0}" -ge "${MINFREE:-44000}" ] || { echo "ABORT gpu $GPU has ${free:-?} MiB free"; exit 1; }
cd "$WT" || exit 1
if [ ! -f "$DUMP/TRAIN_DONE" ]; then
  [ -e "$DUMP" ] && { echo "ABORT $DUMP exists without TRAIN_DONE (an unfinished run?): move it away first"; exit 1; }
  mkdir -p "$DUMP"
  export CUDA_VISIBLE_DEVICES=$GPU
  export OURO_SFT_MIX=tb80sft OURO_SFT_LOCAL_JSONL=$DATA
  unset OURO_INIT_FROM OURO_INIT_DCP OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS OURO_SFT_PROSE_KEEP OURO_OPT_PROBE OURO_SFT_DATA_SEED
  export WANDB_MODE=offline WANDB_PROJECT=ouro-tb WANDB_RUN_NAME=$TAG WANDB_RUN_GROUP=from-base
  export WANDB_DIR=$S/wandb
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export OURO_VAL_FILE=$VAL OURO_VAL_EVERY=${VAL_EVERY:-1000} OURO_VAL_OUT=$S/lr_debug/valprobe_${TAG}.jsonl
  : > "$OURO_VAL_OUT"
  echo "=== $TAG: base -> $STEPS steps, lr $LR cosine, gpu $GPU, start $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
  "$PY" -m torch.distributed.run --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 --role rank --tee 1 -m torchtitan.train \
    --module ouro --config ouro_2_6b_thinking_terminal_sft \
    --parallelism.data_parallel_shard_degree 1 \
    --checkpoint.initial_load_path "$BASE" --checkpoint.initial_load_in_hf \
    --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
    --training.steps "$STEPS" --training.seq_len 4096 --optimizer.lr "$LR" \
    --checkpoint.interval "${CKPT_INTERVAL:-5546}" --metrics.log_freq 50 --metrics.enable_wandb \
    > "$LOG" 2>&1
  rc=$?
  echo "training exited $rc $(date '+%m-%d %H:%M'); checkpoints: $(ls "$DUMP/checkpoint" 2>/dev/null | tr '\n' ' ')" | tee -a "$REP"
  [ "$rc" -eq 0 ] && ls "$DUMP/checkpoint/step-$STEPS"/model*.safetensors > /dev/null 2>&1 && touch "$DUMP/TRAIN_DONE"
  unset CUDA_VISIBLE_DEVICES
fi
[ -f "$DUMP/TRAIN_DONE" ] || { echo "FROM-BASE run $TAG did not finish; no evaluation" | tee -a "$REP"; exit 1; }
[ -n "${NO_EVAL:-}" ] && { echo "FROM-BASE $TAG trained; evaluation skipped (NO_EVAL)" | tee -a "$REP"; exit 0; }
src=$DUMP/checkpoint/step-$STEPS
clean=${DUMP}_clean
if ! ls "$clean"/model*.safetensors > /dev/null 2>&1; then
  mkdir -p "$clean"
  cp "$src"/model*.safetensors "$clean"/
  [ -f "$src/model.safetensors.index.json" ] && cp "$src/model.safetensors.index.json" "$clean"/
  for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$clean/$f"; done
fi
echo "staged $clean; Terminal-Bench test evaluation starts $(date '+%m-%d %H:%M')" | tee -a "$REP"
export OURO_SEQ_CAP=12000
GPU=$GPU GATE=0 CKPT="$clean" NAME=${TAG}_eval PORT=8221 ATT=2 MINFREE=${MINFREE:-44000} OUTROOT=/srv/scratch/jli199/tb_runs TASKS="$ALL11" \
  bash eval_gated.sh > "$TMP/${TAG}_eval.log" 2>&1
python3 -I tb_test_summary.py "${TAG}_eval" /srv/scratch/jli199/tb_runs | tee -a "$REP"
echo "=== FROM-BASE $TAG done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
