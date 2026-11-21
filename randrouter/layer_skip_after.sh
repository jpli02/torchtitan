#!/bin/bash
# Run the static sub-network probe (layer_skip_probe.py) on a card once a job of mine that holds it has ended.
#   GPU=9 WAIT_PID=<pid> bash randrouter/layer_skip_after.sh
set -u
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft || exit 1
GPU=${GPU:-9}
if [ -n "${WAIT_PID:-}" ]; then until ! kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; fi
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -n 1)" -ge 8800 ]; do sleep 30; done
echo "layer skip probe starts on card $GPU $(date '+%m-%d %H:%M')"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True /home/jli199/boptim-agent/.venv/bin/python randrouter/layer_skip_probe.py --mem_gb 8.5
echo "layer skip probe exited $? $(date '+%m-%d %H:%M')"
