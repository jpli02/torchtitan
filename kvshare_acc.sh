#!/bin/bash
# Accuracy check of decode-time KV sharing (kv_share_decode) vs the stock cache, K=4, greedy:
# GSM8K first 100 items (stock first-100 = 0.920) and CRUXEval first 100 (stock first-100 = 0.650).
# Waits for a GPU with >= 30 GB free (other than 5).
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
REP=$TMP/kvshare_acc.txt; : > "$REP"
cd "$WT" || exit 1
gpu=""; END=$(( $(date +%s) + 172800 ))
while [ "$(date +%s)" -lt "$END" ]; do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | awk -F, '$1!=5 && $2+0>=30000 {print $2+0, $1}' | sort -rn | head -1 | awk '{print $2}')
  [ -n "$gpu" ] && break; sleep 120
done
[ -n "$gpu" ] || { echo "ABORT no GPU" | tee -a "$REP"; exit 1; }
echo "=== KVSHARE ACC on gpu $gpu $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
CUDA_VISIBLE_DEVICES=$gpu PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$TMP/gsm8k_test.jsonl" --task gsm8k --steps 4 --n 100 --batch 6 --max_new 2048 --cfg kv_share_decode=true --out "$TMP/kvshare_gsm8k_k4.jsonl" > "$TMP/kvshare_gsm8k.log" 2>&1
grep '^RESULT' "$TMP/kvshare_gsm8k.log" | sed 's/^/gsm8k share_decode: /' | tee -a "$REP"
CUDA_VISIBLE_DEVICES=$gpu PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$TMP/cruxeval.jsonl" --task crux --steps 4 --n 100 --batch 6 --max_new 2048 --cfg kv_share_decode=true --out "$TMP/kvshare_crux_k4.jsonl" > "$TMP/kvshare_crux.log" 2>&1
grep '^RESULT' "$TMP/kvshare_crux.log" | sed 's/^/crux share_decode: /' | tee -a "$REP"
echo "reference stock first-100: gsm8k 0.920 (trunc 0.05), crux 0.650 (trunc 0.17)" | tee -a "$REP"
echo "=== KVSHARE ACC done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
