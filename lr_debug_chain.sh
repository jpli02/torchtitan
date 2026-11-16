#!/bin/bash
# Learning-rate debug on GPU 1, as soon as the epoch 8 test evaluation there has ended:
#   1. optimizer probe with the recipe's optimizer (torchao AdamW8bit), 150 real training steps from the epoch 4 checkpoint;
#   2. the same probe with AdamWbf16 (moments in bfloat16), to check that it fits the card and tracks exact AdamW.
# Start detached:  setsid nohup bash lr_debug_chain.sh >> $TMP/lr_debug_chain.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
cd "$WT" || exit 1
END=$(( $(date +%s) + 14400 ))
while pgrep -u "$(id -un)" -f 'epoch_eval_chain.sh|eval_gated.sh|ouro_openai_server.py' > /dev/null \
   || [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | head -n 1)" -lt 44000 ]; do
  [ "$(date +%s)" -gt "$END" ] && { echo "LR-DEBUG gave up waiting for gpu 1 $(date '+%m-%d %H:%M')"; exit 1; }
  sleep 30
done
echo "gpu 1 free $(date '+%m-%d %H:%M')"
GPU=1 TAG=probe_ep4_8bit OPT=AdamW8bit STEPS=${STEPS:-150} bash lr_probe_run.sh
"$PY" -I lr_debug_probe_report.py "$TMP/opt_probe_probe_ep4_8bit.jsonl" 60
grep -c 'step: ' "$TMP/lrdebug_probe_ep4_8bit_train.log"
GPU=1 TAG=probe_ep4_bf16 OPT=AdamWbf16 STEPS=${STEPS:-150} MINFREE=44000 bash lr_probe_run.sh
"$PY" -I lr_debug_probe_report.py "$TMP/opt_probe_probe_ep4_bf16.jsonl" 60
grep 'memory:' "$TMP/lrdebug_probe_ep4_bf16_train.log" | tail -n 1 | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-200
echo "LR-DEBUG probes done $(date '+%m-%d %H:%M')"
