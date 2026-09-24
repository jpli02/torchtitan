#!/bin/bash
# Per-skill SFT probe on Ouro-2.6B: continue from the 2.6B aggregate-SFT agent
# (6/80), fine-tune on a skill-concentrated set (build_skill_sft.py, 40%), then
# eval pass@k on that skill's test tasks. Same design as the 1.4B run_skill.sh
# (which was null) -- now with the larger model that already breaks the ceiling.
#
#   SKILL=perms-arch TASKS="fix-permissions extract-safely ..." STEPS=1000 ATT=2 GPU=0 bash run_skill_26b.sh
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
SKILL=${SKILL:?skill name}
TASKS=${TASKS:?space-separated test tasks}
STEPS=${STEPS:-1000}
ATT=${ATT:-2}
FRAC=${FRAC:-0.40}
TOTAL=${TOTAL:-2000}
SEQLEN=${SEQLEN:-4096}
BASE=${BASE:-/home/jli199/boptim_scratch/ouro_tb80sft26b_v3_clean}
BASEHF=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
JSONL=/home/jli199/boptim_scratch/tb80sft/skill26_${SKILL}.jsonl
DUMP=/home/jli199/boptim_scratch/ouro_skill26_${SKILL}
CLEAN=${DUMP}_clean
OUT=$TMP/skill26_${SKILL}.txt
MINFREE_TRAIN=${MINFREE_TRAIN:-36000}
: > "$OUT"
cd "$WT" || exit 1

/home/jli199/torchtitan/.venv/bin/python build_skill_sft.py --skill "$SKILL" \
  --total "$TOTAL" --frac "$FRAC" --out "$JSONL" | tee -a "$OUT"

gpu="${GPU:-}"
if [ -z "$gpu" ]; then
  for _ in $(seq 1 1440); do
    gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | tr -d ' ' | awk -F, -v m="$MINFREE_TRAIN" '$2+0>=m {print $2+0, $1}' | sort -rn | head -1 | awk '{print $2}')
    [ -n "$gpu" ] && break
    sleep 60
  done
fi
[ -n "$gpu" ] || { echo "ABORT - no GPU" | tee -a "$OUT"; exit 1; }
echo "[skill26 $SKILL] 2.6B from SFT ckpt on gpu $gpu, $STEPS steps $(date '+%m-%d %H:%M')" | tee -a "$OUT"

export CUDA_VISIBLE_DEVICES=$gpu
export OURO_SFT_MIX=tb80sft
export OURO_SFT_LOCAL_JSONL=$JSONL
export OURO_INIT_FROM=$BASE
unset OURO_INIT_DCP OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
export WANDB_MODE=disabled PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$DUMP"

/home/jli199/torchtitan/.venv/bin/torchrun \
  --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 -m torchtitan.train \
  --module ouro --config ouro_2_6b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps "$STEPS" --training.seq_len "$SEQLEN" --checkpoint.interval "$STEPS" --metrics.log_freq 50 \
  > /home/jli199/terminal_bench_eval/logs/skill26_${SKILL}_train.log 2>&1
echo "training exited $(date '+%m-%d %H:%M')" | tee -a "$OUT"

SRC=$(ls -d "$DUMP"/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$SRC" ] || { echo "ABORT - no checkpoint" | tee -a "$OUT"; exit 1; }
rm -rf "$CLEAN"; mkdir -p "$CLEAN"
cp "$SRC"/model*.safetensors "$CLEAN"/ 2>/dev/null
[ -f "$SRC/model.safetensors.index.json" ] && cp "$SRC/model.safetensors.index.json" "$CLEAN"/
for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json \
         tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do
  cp "$BASEHF/$f" "$CLEAN/$f"
done
echo "staged $CLEAN ($(ls "$CLEAN" | wc -l) files)" | tee -a "$OUT"

GPU=$gpu GATE=0 CKPT="$CLEAN" NAME="skill26_${SKILL}" PORT=8150 ATT=$ATT MINFREE=20000 \
  TASKS="$TASKS" bash eval_gated.sh >> "$OUT" 2>&1
echo "--- skill26 $SKILL done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
/home/jli199/torchtitan/.venv/bin/python "$TMP/tally_skill.py" "skill26_${SKILL}" $TASKS | tee -a "$OUT"
