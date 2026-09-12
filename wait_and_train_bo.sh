#!/bin/bash
# Wait for a GPU with >=33GB free (Qwen3-1.7B full FT peaked at 32.7GB in the
# smoke), then launch the BO-agent SFT on the full 3,915-row set for 1000 steps.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
OUT=/home/jli199/.claude/jobs/c0d2da0a/tmp/wait_and_train_bo.txt
: > "$OUT"
cd "$WT" || exit 1
gpu=""
for _ in $(seq 1 720); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, '$2+0>=33000 {print $1; exit}')
  [ -n "$gpu" ] && break
  sleep 60
done
[ -n "$gpu" ] || { echo "ABORT - no GPU with 33GB free after 12h" | tee -a "$OUT"; exit 1; }
echo "[bo-sft] GPU $gpu free, launching $(date '+%m-%d %H:%M')" | tee -a "$OUT"
TAG=bo_v1 STEPS=1000 GPU=$gpu WANDB_MODE=offline \
  JSONL=/home/jli199/boptim_scratch/bo_sft/train.jsonl bash run_bo_sft.sh
echo "[bo-sft] run_bo_sft returned $(date '+%m-%d %H:%M')" | tee -a "$OUT"
tail -6 /home/jli199/.claude/jobs/c0d2da0a/tmp/qwen_bo_v1.txt 2>/dev/null | tee -a "$OUT"
