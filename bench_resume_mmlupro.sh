#!/bin/bash
# The bench queue grabbed GPU 5 while the TB eval server there was still small, so the
# MMLU-Pro K=2 run and the guard eval now share that card. Let K=2 finish (it is at 258/300),
# then stop the queue/pipeline shells and run the remaining MMLU-Pro points and the SuperGPQA
# pipeline on GPU 4 (free) instead.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
END=$(( $(date +%s) + 14400 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q '^RESULT k2' "$TMP/bench_mmlupro.txt" 2>/dev/null && break
  sleep 60
done
for p in $(pgrep -f 'bash bench_queue.sh'); do kill "$p" 2>/dev/null; done
for p in $(pgrep -f 'bash bench_pipeline.sh'); do kill "$p" 2>/dev/null; done
sleep 3
for p in $(pgrep -f 'depth_eval.py .*mmlupro.jsonl'); do kill "$p" 2>/dev/null && echo "killed stray mmlupro eval $p"; done
echo "bench queue moved off GPU 5 $(date '+%m-%d %H:%M'); remaining MMLU-Pro points + SuperGPQA on GPU 4" | tee -a "$TMP/bench_mmlupro.txt"
TASK=mc DATA=$TMP/mmlupro.jsonl N=300 THRS="0.3 0.1,0.5,0.5" GPU=4 bash gate_sweep.sh
cp "$TMP/gate_mc.txt" "$TMP/gate_mmlupro.txt" 2>/dev/null
NAME=supergpqa TASK=mc DATA=$TMP/supergpqa.jsonl N=300 GPU=4 MAXNEW=2048 bash bench_pipeline.sh
echo "bench resume done $(date '+%m-%d %H:%M')"
