#!/bin/bash
# After the 20k-step TerminalTraj SFT: stage the final checkpoint and eval on
# the SAME 12 tasks, UNGATED (stock terminus-2), so the number sits directly
# beside continue10k 5/24, batching5k 4/24, longhorizon1k 3/24, traj1k 3/24.
#
# MINFREE=35000, not 20000: the traj1k eval picked a 23.6GB card, the KV cache
# outgrew it over long agent sessions, and 278 CUDA OOMs voided fix-permissions
# outright. A 12-task run at 2 attempts needs real headroom.
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
DUMP=/home/jli199/boptim_scratch/ouro_traj20k
CLEAN=/home/jli199/boptim_scratch/ouro_traj20k_clean
OUT=$TMP/traj20k_chain.txt
: "${TRAIN_PID:?TRAIN_PID must be set}"
: > "$OUT"

# Measured 6.9 s/step at step 100 (slower than the batching run's 3.3-5.3),
# so 20k steps is ~38h. WAIT_MIN caps the wait; default 3000 min = 50h.
for _ in $(seq 1 "${WAIT_MIN:-3000}"); do
  kill -0 "$TRAIN_PID" 2>/dev/null || break
  sleep 60
done
if kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "ABORT - training still alive after 36h" | tee -a "$OUT"; exit 1
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
for _ in $(seq 1 240); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, '$2+0>=35000 {print $1; exit}')
  [ -n "$gpu" ] && break
  sleep 60
done
[ -n "$gpu" ] || { echo "ABORT - no GPU with 35GB free after 4h" | tee -a "$OUT"; exit 1; }

GPU=$gpu GATE=0 CKPT="$CLEAN" NAME=traj20k PORT=8097 ATT=2 MINFREE=35000 \
  bash eval_gated.sh >> "$OUT" 2>&1
echo "--- chain done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
