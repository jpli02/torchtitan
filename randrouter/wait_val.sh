#!/bin/bash
# Wait until every listed training run has logged its validation at the given step (or one of them failed), then print the
# validation blocks.   bash randrouter/wait_val.sh STEP tag1 [tag2 ...]
R=/home/jli199/boptim_scratch/randrouter
step=$1
shift
while true; do
  all=1
  for t in "$@"; do
    grep -q "val step $step\]" "$R/$t.log" 2>/dev/null || all=0
    if grep -q "Traceback" "$R/$t.log" 2>/dev/null; then echo "FAILED $t"; tail -n 3 "$R/$t.log" | cut -c1-300; all=2; fi
  done
  [ "$all" != 0 ] && break
  sleep 30
done
date '+%H:%M:%S'
for t in "$@"; do
  echo "== $t"
  grep "^step" "$R/$t.log" | tail -n 2 | cut -c1-240
  grep -v "Warning\|deprecated\|Consider using\|agg\[" "$R/$t.log" | grep -B8 "val step $step\]" | cut -c1-240
done
