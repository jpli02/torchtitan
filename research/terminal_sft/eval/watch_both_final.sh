#!/bin/bash
# Wait for both outstanding evals: repro_c10k on 12 tasks, continue10k on 80.
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
while true; do
  r_done=$(grep -c -- "--- done ---" "$TMP/repro12_results.txt" 2>/dev/null)
  c_done=$(grep -c -- "--- C10K 80-TASK DONE ---" "$TMP/c10k80_state.txt" 2>/dev/null)
  r_n=$(ls /tmp/tb_runs/clean_repro11k/*/*/results.json 2>/dev/null | wc -l)
  c_n=$(ls /tmp/tb_runs/e80_c10k80_s*/*/*/results.json 2>/dev/null | wc -l)
  if [ "$r_done" -gt 0 ] && [ "$c_done" -gt 0 ]; then
    echo "BOTH DONE  repro12=$r_n/24  c10k80=$c_n/80"; break
  fi
  r_up=$(pgrep -f research/terminal_sft/eval/eval_repro12.sh >/dev/null 2>&1 && echo 1 || echo 0)
  c_up=$(pgrep -f research/terminal_sft/eval/run_c10k_80.sh >/dev/null 2>&1 && echo 1 || echo 0)
  if [ "$r_up" -eq 0 ] && [ "$c_up" -eq 0 ]; then
    echo "BOTH EXITED  repro12=$r_n/24 (done=$r_done)  c10k80=$c_n/80 (done=$c_done)"; break
  fi
  sleep 600
done
