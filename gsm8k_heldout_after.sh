#!/bin/bash
# After the GPU-4 per-step CRUX run finishes, run the GSM8K held-out validation there
# (items 300-599): full depth, released gate q=0.3, per-step rule (0.1,0.5,0.5).
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
busy() { for p in $(pgrep -f 'depth_eval.py'); do tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q '^CUDA_VISIBLE_DEVICES=4$' && return 0; done; return 1; }
END=$(( $(date +%s) + 86400 ))
while [ "$(date +%s)" -lt "$END" ] && busy; do sleep 120; done
echo "GPU 4 free; GSM8K held-out start $(date '+%m-%d %H:%M')"
TASK=gsm8k DATA=$TMP/gsm8k_test.jsonl N=300 OFFSET=300 THRS="none 0.3 0.1,0.5,0.5" GPU=4 bash gate_sweep.sh
