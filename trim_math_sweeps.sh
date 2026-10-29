#!/bin/bash
# Trim the GSM8K (GPU 3) and MATH-500 (GPU 5) depth sweeps to K in {4,2,1}.
# Kill both chain shells now (their running depth_eval children finish on their own),
# then per GPU wait for the running eval and run only the depths still missing.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
pkill -f 'bash depth_sweep.sh' 2>/dev/null
pkill -f 'bash gpu5_after_ep2.sh' 2>/dev/null
echo "chains killed $(date '+%m-%d %H:%M'); running evals: $(pgrep -af 'depth_eval.py' | grep -oE 'task [a-z0-9]+ --steps [0-9]+' | tr '\n' ';')"

missing() {  # missing <task> -> depths among 4 2 1 without a RESULT line
  local t=$1 out=""
  for K in 4 2 1; do grep -q "\"task\": \"$t\", \"steps\": $K," "$TMP/depth_$t.txt" 2>/dev/null || out="$out $K"; done
  echo "$out"
}
(
  while pgrep -f 'depth_eval.py .*--task gsm8k' >/dev/null; do sleep 60; done
  left=$(missing gsm8k)
  echo "gsm8k: remaining depths:$left $(date '+%m-%d %H:%M')"
  [ -n "$left" ] && TASK=gsm8k DATA=$TMP/gsm8k_test.jsonl N=300 STEPS="$left" BATCH=6 MAXNEW=2048 GPU=3 bash depth_sweep.sh
  echo "gsm8k trimmed sweep done $(date '+%m-%d %H:%M')"
) &
(
  while pgrep -f 'depth_eval.py .*--task math' >/dev/null; do sleep 60; done
  left=$(missing math)
  echo "math: remaining depths:$left $(date '+%m-%d %H:%M')"
  [ -n "$left" ] && TASK=math DATA=$TMP/math500.jsonl N=150 STEPS="$left" BATCH=4 MAXNEW=3072 GPU=5 bash depth_sweep.sh
  echo "math trimmed sweep done $(date '+%m-%d %H:%M')"
) &
wait
echo "TRIM done $(date '+%m-%d %H:%M')"
