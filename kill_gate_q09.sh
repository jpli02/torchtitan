#!/bin/bash
# Stop the gate-sweep q=0.9 evaluation (my own job; the chain shell was already stopped).
for p in $(pgrep -f 'depth_eval.py .*--threshold 0.9'); do kill "$p" 2>/dev/null && echo "killed gate q=0.9 eval pid $p"; done
sleep 2
echo "remaining threshold evals: $(pgrep -af 'depth_eval.py .*--threshold' | grep -oE 'threshold [0-9.]+' | tr '\n' ' ')"
