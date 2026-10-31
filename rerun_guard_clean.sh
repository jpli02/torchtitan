#!/bin/bash
# Clean re-run of the easy-tier stock-vs-guard eval (the first run shared GPU 5 with an
# MMLU-Pro sweep from 06:20 on 10-31). Waits until the current guard eval has finished and
# GPU 2 is free, then runs both arms there under RUNTAG=clean.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
END=$(( $(date +%s) + 172800 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q 'GUARD EVAL done' "$TMP/eval_guard_easy.txt" 2>/dev/null || { sleep 300; continue; }
  busy=0; for p in $(pgrep -f 'depth_eval.py'); do tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q '^CUDA_VISIBLE_DEVICES=2$' && busy=1; done
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 | tr -d ' ')
  [ "$busy" -eq 0 ] && [ "${free:-0}" -ge 40000 ] && break
  sleep 300
done
echo "clean guard re-run starting on GPU 2 $(date '+%m-%d %H:%M')"
GPU=2 RUNTAG=clean bash eval_guard_easy.sh
