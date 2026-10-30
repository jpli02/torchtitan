#!/bin/bash
# Stop the stray CRUXEval beyond-depth-4 runs (my own evals) and their chain shells.
for p in $(pgrep -f 'depth_eval.py .*--task crux --steps [58]'); do kill "$p" 2>/dev/null && echo "killed depth_eval crux K>4 pid $p"; done
for p in $(pgrep -f 'bash depth_sweep.sh'); do
  if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q '^TASK=crux$'; then kill "$p" 2>/dev/null && echo "killed crux depth_sweep chain pid $p"; fi
done
sleep 3
echo "remaining crux evals: $(pgrep -af 'depth_eval.py .*--task crux' | grep -oE 'steps [0-9]+' | tr '\n' ' ')"
