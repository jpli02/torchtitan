#!/bin/bash
# One-shot: ~25 min after training started, confirm it is actually STEPPING
# (loss lines growing) and the wandb curve is populating, i.e. not hung in compile.
L=/home/jli199/terminal_bench_eval/logs/ouro_tvary_r1_train.log
END=$(( $(date +%s) + 1500 )); while [ "$(date +%s)" -lt "$END" ]; do sleep 60; done
echo "=== round-1 stepping check $(date '+%m-%d %H:%M') ==="
echo "torchtitan alive: $(pgrep -f 'torchtitan.train' | grep -v pgrep | grep -c .)"
n=$(grep -c 'loss:' "$L" 2>/dev/null)
echo "loss lines: $n   first/latest: $(grep -oE 'step: *[0-9]+ *loss: *[0-9.]+' "$L" | sed 's/\x1b\[[0-9;]*m//g' | sed -n '1p;$p' | tr '\n' ' ')"
echo "tracebacks: $(grep -c Traceback "$L" 2>/dev/null)"
if [ "$n" -eq 0 ]; then echo "WARNING: no loss lines after ~25 min -> possible compile hang; inspect $L"; fi
