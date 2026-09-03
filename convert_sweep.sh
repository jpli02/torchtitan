#!/bin/bash
# Convert the intermediate batching-SFT checkpoints (DCP) to servable HF dirs.
#
# Why: batching5k = continue10k + 5k steps over a corpus the MIN_CMDS=1.5 filter
# cut to 36% of its rows. Heavy repetition of a narrow slice gains the target
# behaviour early and forgets general capability late, which is exactly the
# shape we measured -- structure improved (timeouts 17->14, multi-command turns
# 0->48) while the marginal task was lost (5/24 -> 4/24).
#
# If that is the mechanism, an EARLY checkpoint holds the gain without the
# damage. All five survived, so testing it costs evals only, no training.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
SRC=/home/jli199/boptim_scratch/ouro_batching_sft/checkpoint
DST=/home/jli199/boptim_scratch/ouro_sweep
TMPL=/home/jli199/torchtitan/assets/hf/Ouro-1.4B-Thinking
mkdir -p "$DST"
cd "$WT" || exit 1

for s in "$@"; do
  out="$DST/step$s"
  if ls "$out"/model*.safetensors >/dev/null 2>&1; then
    echo "step-$s already converted"; continue
  fi
  echo "=== converting step-$s ($(date '+%H:%M')) ==="
  CUDA_VISIBLE_DEVICES="${CVD:-4}" PYTHONPATH="$WT" /home/jli199/torchtitan/.venv/bin/python \
    scripts/export_dcp_to_hf.py \
    --module ouro --config ouro_1_4b_thinking_terminal_sft \
    --checkpoint "$SRC/step-$s" --hf_template "$TMPL" \
    --out "$out" --dtype bf16 2>&1 | tail -3
  # the serving loader hangs on a sharded/ subdir; make sure none rides along
  rm -rf "$out/sharded"
  ls "$out"/model*.safetensors >/dev/null 2>&1 \
    && echo "step-$s OK" || echo "step-$s FAILED"
done
