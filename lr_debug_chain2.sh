#!/bin/bash
# Learning-rate debug on GPU 1, part 2:
#   0. validation loss of the two checkpoints that ran out of memory in the first batch (r1, ep5), two at a time
#      (one fit evaluation peaks at 11.9 GB, so four in parallel do not fit a 48 GB card);
#   then constant-LR runs from v3 on the r1 mix with held-out validation every 134 steps, so that learning rates are
#   compared on validation curves instead of Terminal-Bench trials:
#   A. lr 5e-6 constant, 1072 steps (2 epochs)      B. lr 2e-5 constant (the recipe's peak), 1072 steps
# Start detached:  setsid nohup bash lr_debug_chain2.sh >> $TMP/lr_debug_chain2.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
S=/home/jli199/boptim_scratch
ARMS=${ARMS:-5e-6 2e-5}
cd "$WT" || exit 1
END=$(( $(date +%s) + 21600 ))
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | head -n 1)" -ge 44000 ]; do
  [ "$(date +%s)" -gt "$END" ] && { echo "LR-DEBUG sweep gave up waiting $(date '+%m-%d %H:%M')"; exit 1; }
  sleep 30
done
GPU=1 PAR=2 bash lr_debug_fit.sh r1=$S/ouro_tvary_r1_clean ep5=$S/ouro_tvary_r1_8ep_ep5_clean
echo "LR-DEBUG checkpoint table complete $(date '+%m-%d %H:%M')"
for arm in $ARMS; do
  GPU=1 LR=$arm STEPS=${STEPS:-1072} TAG=lr${arm}_const bash lr_sweep_run.sh
  if [ "$(wc -l < "$TMP/valprobe_lr${arm}_const.jsonl")" -lt 5 ]; then
    echo "LR-DEBUG sweep stopped: run lr${arm}_const wrote no validation rows $(date '+%m-%d %H:%M')"
    tail -n 30 "/home/jli199/terminal_bench_eval/logs/ouro_lrsweep_lr${arm}_const_train.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-220
    exit 1
  fi
  "$PY" -I lr_debug_valcurve.py "lr${arm}_const"
done
echo "LR-DEBUG sweep part 1 done $(date '+%m-%d %H:%M')"
