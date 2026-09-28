#!/bin/bash
# After the TerminalTraj 1k SFT: stage a servable copy and eval on the SAME 12
# tasks, UNGATED (stock terminus-2), so the number is comparable to the existing
# baselines -- continue10k 5/24, batching5k 4/24, longhorizon1k 3/24.
#
# Ungated is also the right test for the behavioural question this run exists to
# answer: does a model trained only on FINISHED trajectories still declare
# task_complete while failing? The verification gate would mask exactly that.
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
DUMP=/home/jli199/boptim_scratch/ouro_traj_sft
CLEAN=/home/jli199/boptim_scratch/ouro_traj_clean
OUT=$TMP/traj12_chain.txt
: "${TRAIN_PID:?TRAIN_PID must be set}"
: > "$OUT"

for _ in $(seq 1 360); do
  kill -0 "$TRAIN_PID" 2>/dev/null || break
  sleep 60
done
if kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "ABORT - training still alive after 6h" | tee -a "$OUT"; exit 1
fi
echo "training exited $(date '+%m-%d %H:%M')" | tee -a "$OUT"

SRC=$(ls -d "$DUMP"/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$SRC" ] || { echo "ABORT - no checkpoint" | tee -a "$OUT"; exit 1; }
echo "checkpoint: $SRC" | tee -a "$OUT"

cd "$WT" || exit 1
/home/jli199/torchtitan/.venv/bin/python scripts/push_ouro_sft_to_hub.py \
  --checkpoint "$SRC" --base_model /home/jli199/torchtitan/assets/hf/Ouro-1.4B-Thinking \
  --repo_id jaslee/tmp --dry_run >/dev/null 2>&1
rm -rf "$CLEAN"; mkdir -p "$CLEAN"
for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json \
         tokenizer_config.json special_tokens_map.json vocab.json merges.txt \
         model.safetensors.index.json model-00001-of-00001.safetensors; do
  [ -e "$SRC/$f" ] && ln -s "$SRC/$f" "$CLEAN/$f"
done
echo "staged $CLEAN" | tee -a "$OUT"

gpu=""
for _ in $(seq 1 120); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, '$2+0>=20000 {print $1; exit}')
  [ -n "$gpu" ] && break
  sleep 60
done
[ -n "$gpu" ] || { echo "ABORT - no free GPU" | tee -a "$OUT"; exit 1; }

GPU=$gpu GATE=0 CKPT="$CLEAN" NAME=traj1k PORT=8095 ATT=2 MINFREE=20000 \
  bash eval_gated.sh >> "$OUT" 2>&1
echo "--- chain done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
