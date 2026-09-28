#!/bin/bash
# HELD-OUT oracle experiment: train on oracle solutions of the 68 tasks NOT in
# the 12-task eval set, then eval the 12. Clean for the 12-task eval;
# contaminated for the 80-task one -- label results accordingly.
#
# oracle12 set the ceiling (train on the 12, eval the 12): effectively 12/12.
# This asks the real question: do oracle-style demonstrations TRANSFER to
# unseen tasks of the same distribution? The comparison points on the 12:
#   continue10k 5/24 (best real)   oracle12 20/24 (memorised ceiling)
#
# Same init (continue10k), same LR, same row construction as oracle12 so the
# only variable is which tasks the demonstrations came from.
#
# 68 rows at ~1.5k tokens is ~25 windows/epoch; 500 steps is ~20 epochs.
# Checkpoint at 250 too, so an earlier point can be evaluated if 500 overfits
# the 68 in a way that hurts transfer.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
DUMP=/home/jli199/boptim_scratch/ouro_oracle68
CLEAN=/home/jli199/boptim_scratch/ouro_oracle68_clean
OUT=$TMP/oracle68_chain.txt
STEPS=${STEPS:-500}
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
echo "[oracle68] training on gpu $gpu, $STEPS steps $(date '+%m-%d %H:%M')" | tee -a "$OUT"

export CUDA_VISIBLE_DEVICES=$gpu
export OURO_SFT_MIX=oracle12          # generic: repo "json" + local JSONL
export OURO_SFT_LOCAL_JSONL=/home/jli199/boptim_scratch/oracle68/train.jsonl
export OURO_INIT_FROM=/home/jli199/boptim_scratch/ouro_c10k_clean
unset OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
export WANDB_MODE=online WANDB_PROJECT=ouro-terminal-sft
export WANDB_RUN_NAME="ouro-oracle68-heldout"
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
  --checkpoint.interval 250 --metrics.log_freq 10 \
  > /home/jli199/terminal_bench_eval/logs/oracle68_train.log 2>&1
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

GPU=$gpu GATE=0 CKPT="$CLEAN" NAME=oracle68 PORT=8100 ATT=2 MINFREE=28000 \
  bash eval_gated.sh >> "$OUT" 2>&1
echo "--- chain done $(date '+%m-%d %H:%M') --- LABEL: held-out oracle, clean for 12-task eval" | tee -a "$OUT"
