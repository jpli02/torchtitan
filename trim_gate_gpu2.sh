#!/bin/bash
# The GPU-2 gate chain was launched with 4 points (0.3, 0.1, then two per-step settings);
# the per-step points now run on GPU 4. Stop the GPU-2 chain once its q=0.1 result exists.
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
END=$(( $(date +%s) + 86400 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q '"threshold": 0.1,' "$TMP/depth_crux.txt" "$TMP/gate_crux.txt" 2>/dev/null && break
  sleep 180
done
for p in $(pgrep -f 'bash gate_sweep.sh'); do
  if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q '^GPU=2$'; then kill "$p" 2>/dev/null && echo "killed GPU-2 gate chain $p"; fi
done
sleep 3
for p in $(pgrep -f 'depth_eval.py .*--threshold_steps'); do
  if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q '^CUDA_VISIBLE_DEVICES=2$'; then kill "$p" 2>/dev/null && echo "killed duplicate per-step eval on GPU 2 ($p)"; fi
done
echo "GPU-2 gate chain trimmed after q=0.1 $(date '+%m-%d %H:%M')" | tee -a "$TMP/gate_crux.txt"
