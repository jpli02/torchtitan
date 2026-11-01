#!/bin/bash
# SuperGPQA is budget-bound at 2,048 tokens (59% truncated, 19% accuracy): stop its pipeline.
for p in $(pgrep -f 'bash bench_resume_mmlupro.sh'); do kill "$p" 2>/dev/null && echo "killed resume chain $p"; done
for p in $(pgrep -f 'bash bench_pipeline.sh'); do kill "$p" 2>/dev/null && echo "killed bench_pipeline $p"; done
sleep 2
for p in $(pgrep -f 'depth_eval.py .*supergpqa.jsonl'); do kill "$p" 2>/dev/null && echo "killed supergpqa eval $p"; done
sleep 3
echo "GPU 4 now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 4)"
