#!/bin/bash
# From-base validation curve without new training: the run that produced v3 (base Ouro-2.6B-Thinking -> 20,000 steps on
# tb80sft/train_v3.jsonl at the recipe's 2e-5 with cosine decay) left checkpoints at steps 5000, 10000 and 15000. Export
# them and score base, 5k, 10k, 15k on the held-out validation file (v3 = step 20000 is already scored), so the question
# "is the learning rate right when training starts from the base model" has data before any new 36-hour run.
#   GPU=0 bash lr_debug_frombase_curve.sh
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
S=/home/jli199/boptim_scratch
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
GPU=${GPU:-0}
cd "$WT" || exit 1
args="base=$BASE"
for step in 5000 10000 15000; do
  src=$S/ouro_tb80sft26b_v3/checkpoint/step-$step
  clean=$S/ouro_tb80sft26b_v3_s${step}_clean
  if ! ls "$clean"/model*.safetensors > /dev/null 2>&1; then
    [ -d "$src" ] || { echo "no checkpoint at $src"; continue; }
    mkdir -p "$clean"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$WT" "$PY" scripts/export_dcp_to_hf.py --module ouro --config ouro_2_6b_thinking_terminal_sft \
      --checkpoint "$src" --hf_template "$BASE" --out "$clean" --dtype bf16 > "$TMP/export_v3_s${step}.log" 2>&1
    echo "export of step $step returned $? $(date '+%m-%d %H:%M')"
    [ -d "$clean/sharded" ] && rm -rf "$clean/sharded"
    for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$clean/$f"; done
  fi
  ls "$clean"/model*.safetensors > /dev/null 2>&1 && args="$args v3s${step}=$clean"
done
GPU=$GPU PAR=2 bash lr_debug_fit.sh $args
echo "FROM-BASE curve done $(date '+%m-%d %H:%M')"
