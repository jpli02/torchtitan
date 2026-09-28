#!/bin/bash
# Per-skill SFT probe: continue from the competent v3 agent, fine-tune on a
# skill-concentrated set ($WT/research/terminal_sft/data/build_skill_sft.py), then eval pass@k on the test
# tasks that need that skill. Isolates the "matched skill data" effect on top
# of a fixed agent baseline (v3 solves 0 of the hard skill tasks).
#
#   SKILL=tmux TASKS="tmux-advanced-workflow git-multibranch" STEPS=1500 ATT=4 bash $WT/research/terminal_sft/train/run_skill.sh
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
SKILL=${SKILL:?skill name}
TASKS=${TASKS:?space-separated test tasks}
STEPS=${STEPS:-1500}
ATT=${ATT:-4}
FRAC=${FRAC:-0.40}
TOTAL=${TOTAL:-2000}
BASE=${BASE:-/home/jli199/boptim_scratch/ouro_tb80sft_v3_clean}
JSONL=/home/jli199/boptim_scratch/tb80sft/skill_${SKILL}.jsonl
DUMP=/home/jli199/boptim_scratch/ouro_skill_${SKILL}
CLEAN=${DUMP}_clean
OUT=$TMP/skill_${SKILL}.txt
: > "$OUT"
cd "$WT" || exit 1

/home/jli199/torchtitan/.venv/bin/python $WT/research/terminal_sft/data/build_skill_sft.py --skill "$SKILL" \
  --total "$TOTAL" --frac "$FRAC" --out "$JSONL" | tee -a "$OUT"

SEQLEN=${SEQLEN:-4096}
MINFREE_TRAIN=${MINFREE_TRAIN:-31000}
gpu="${GPU:-}"   # GPU=<n> pins the card; else auto-pick the FREEST card >= MINFREE_TRAIN
if [ -z "$gpu" ]; then
  for _ in $(seq 1 720); do
    gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | tr -d ' ' | awk -F, -v m="$MINFREE_TRAIN" '$2+0>=m {print $2+0, $1}' | sort -rn | head -1 | awk '{print $2}')
    [ -n "$gpu" ] && break
    sleep 60
  done
fi
[ -n "$gpu" ] || { echo "ABORT - no GPU" | tee -a "$OUT"; exit 1; }
echo "[skill $SKILL] train from v3 on gpu $gpu, $STEPS steps $(date '+%m-%d %H:%M')" | tee -a "$OUT"

export CUDA_VISIBLE_DEVICES=$gpu
export OURO_SFT_MIX=tb80sft
export OURO_SFT_LOCAL_JSONL=$JSONL
export OURO_INIT_FROM=$BASE
unset OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
export WANDB_MODE=disabled PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$DUMP"

/home/jli199/torchtitan/.venv/bin/torchrun \
  --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 -m torchtitan.train \
  --module ouro --config ouro_1_4b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps "$STEPS" --training.seq_len "$SEQLEN" --checkpoint.interval "$STEPS" --metrics.log_freq 50 \
  > /home/jli199/terminal_bench_eval/logs/skill_${SKILL}_train.log 2>&1
echo "training exited $(date '+%m-%d %H:%M')" | tee -a "$OUT"

SRC=$(ls -d "$DUMP"/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$SRC" ] || { echo "ABORT - no checkpoint" | tee -a "$OUT"; exit 1; }
BASEHF=/home/jli199/torchtitan/assets/hf/Ouro-1.4B-Thinking
rm -rf "$CLEAN"; mkdir -p "$CLEAN"
cp "$SRC"/model-00001-of-00001.safetensors "$SRC"/model.safetensors.index.json "$CLEAN"/
for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json \
         tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do
  cp "$BASEHF/$f" "$CLEAN/$f"
done
echo "staged $CLEAN ($(ls "$CLEAN" | wc -l) files)" | tee -a "$OUT"

GPU=$gpu GATE=0 CKPT="$CLEAN" NAME="skill_${SKILL}" PORT=8140 ATT=$ATT MINFREE=28000 \
  TASKS="$TASKS" bash $WT/research/terminal_sft/eval/eval_gated.sh >> "$OUT" 2>&1
echo "--- skill $SKILL done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
/home/jli199/torchtitan/.venv/bin/python "$TMP/tally_skill.py" "skill_${SKILL}" $TASKS | tee -a "$OUT"
