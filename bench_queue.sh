#!/bin/bash
# Run the router-headroom pipeline on MMLU-Pro then SuperGPQA on the first free card.
# Waits until the GSM8K held-out chain has claimed GPU 4 (avoids racing it), then polls.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
while ! grep -q 'GATE SWEEP gsm8k' "$TMP/gate_gsm8k.txt" 2>/dev/null; do sleep 120; done
gpu=""; END=$(( $(date +%s) + 259200 ))
while [ "$(date +%s)" -lt "$END" ]; do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | awk -F, '$2+0>=40000 {print $2+0, $1}' | sort -rn | head -1 | awk '{print $2}')
  [ -n "$gpu" ] && break; sleep 180
done
[ -n "$gpu" ] || { echo "ABORT no GPU in 72h"; exit 1; }
echo "bench queue on gpu $gpu $(date '+%m-%d %H:%M')"
NAME=mmlupro TASK=mc DATA=$TMP/mmlupro.jsonl N=300 GPU=$gpu MAXNEW=2048 bash bench_pipeline.sh
NAME=supergpqa TASK=mc DATA=$TMP/supergpqa.jsonl N=300 GPU=$gpu MAXNEW=2048 bash bench_pipeline.sh
echo "bench queue done $(date '+%m-%d %H:%M')"
