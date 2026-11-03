#!/bin/bash
# Wait until the GPU 1 LiveCodeBench chain (lcb_pipeline.sh) has graded K=2, then stop it (its remaining points run
# on GPU 3 via lcb_tail_gpu3.sh) and start the GSM-Symbolic "main" pipeline on the freed GPU 1.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
REP=$TMP/bench_lcb342.txt
END=$(( $(date +%s) + 43200 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q '^k2 GRADED' "$REP" 2>/dev/null && break
  sleep 60
done
grep -q '^k2 GRADED' "$REP" 2>/dev/null || { echo "k2 never graded; not stopping the chain $(date '+%m-%d %H:%M')"; exit 1; }
for p in $(pgrep -f 'bash lcb_pipeline.sh'); do
  ch=$(pgrep -P "$p" | tr '\n' ' ')
  kill "$p" 2>/dev/null && echo "stopped chain shell $p (children: $ch)"
  for c in $ch; do kill "$c" 2>/dev/null; done
done
sleep 10
pgrep -af 'bench_lcb342_(q0.5|q0.3|steps|k1)' | grep -v 'lcb_tail_gpu3\|CUDA_VISIBLE_DEVICES=3' | cut -c1-120
echo "LCB GPU 1 chain stopped after K=2 $(date '+%m-%d %H:%M')" | tee -a "$REP"
cd "$WT" || exit 1
NAME=gsmsym_main TASK=gsm8k DATA=$TMP/gsm_symbolic_main.jsonl N=300 GPU=1 MAXNEW=2048 BATCH=6 bash bench_pipeline.sh > "$TMP/gsmsym_main_driver.log" 2>&1
echo "gsmsym_main pipeline finished $(date '+%m-%d %H:%M')"
