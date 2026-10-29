#!/bin/bash
# Coding depth sweeps, queued behind the trimmed math sweeps:
#   GPU 3 (after GSM8K K=1): CRUXEval output prediction, 200 items, K in 4 2 1 6, max_new 2048
#   GPU 5 (after MATH K=1):  LiveCodeBench easy+medium, 120 problems, K in 4 2 1 6, max_new 3072,
#                            each depth graded in a docker sandbox right after generation
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1

wait_task_done() {  # wait_task_done <task> : K=1 RESULT present and no eval of that task running
  local t=$1 END=$(( $(date +%s) + 86400 ))
  while [ "$(date +%s)" -lt "$END" ]; do
    if grep -q "\"task\": \"$t\", \"steps\": 1," "$TMP/depth_$t.txt" 2>/dev/null && ! pgrep -f "depth_eval.py .*--task $t" >/dev/null; then return 0; fi
    sleep 120
  done
  return 1
}

(
  wait_task_done gsm8k || echo "crux: gsm8k never finished; starting anyway $(date '+%m-%d %H:%M')"
  echo "crux sweep start $(date '+%m-%d %H:%M')"
  TASK=crux DATA=$TMP/cruxeval.jsonl N=200 STEPS="4 2 1 6" BATCH=6 MAXNEW=2048 GPU=3 bash depth_sweep.sh
) > "$TMP/coding_gpu3.log" 2>&1 &

(
  wait_task_done math || echo "lcb: math never finished; starting anyway $(date '+%m-%d %H:%M')"
  echo "lcb sweep start $(date '+%m-%d %H:%M')"
  cp "$WT/lcb_grade.py" "$TMP/lcb_grade.py"
  for K in 4 2 1 6; do
    TASK=lcb DATA=$TMP/lcb_em120.jsonl N=120 STEPS="$K" BATCH=4 MAXNEW=3072 GPU=5 bash depth_sweep.sh
    docker run --rm --network none --memory 4g --cpus 4 -v "$TMP":/w python:3.12-slim \
      python /w/lcb_grade.py /w/lcb_em120.jsonl "/w/depth_lcb_k${K}.jsonl" "/w/depth_lcb_k${K}.graded.json" 2>&1 | grep -E '^GRADED' | sed "s/^/K=$K /" | tee -a "$TMP/depth_lcb.txt"
  done
  echo "lcb sweep done $(date '+%m-%d %H:%M')"
) > "$TMP/coding_gpu5.log" 2>&1 &
wait
echo "CODING SWEEPS done $(date '+%m-%d %H:%M')"
