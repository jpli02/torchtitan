#!/bin/bash
# Drop the beyond-trained-depth runs (Ouro trained at T=4): keep CRUX K=3, cancel K=5/8;
# keep LCB K=4,2,1 then run K=3, cancel K=6.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
done_c=0; done_l=0
END=$(( $(date +%s) + 172800 ))
while [ "$(date +%s)" -lt "$END" ] && { [ "$done_c" -eq 0 ] || [ "$done_l" -eq 0 ]; }; do
  if [ "$done_c" -eq 0 ] && grep -q '"task": "crux", "steps": 3,' "$TMP/depth_crux.txt" 2>/dev/null; then
    pkill -f 'bash crux_358_poll.sh' 2>/dev/null
    pkill -f 'depth_sweep.sh' -P "$(pgrep -f 'bash crux_358_poll.sh' | head -1)" 2>/dev/null
    for p in $(pgrep -f 'depth_eval.py .*--task crux --steps [58]'); do kill "$p" 2>/dev/null; done
    for p in $(pgrep -f 'bash depth_sweep.sh'); do grep -q 'TASK=crux' "/proc/$p/environ" 2>/dev/null && kill "$p" 2>/dev/null; done
    echo "crux: K=3 done, K=5/8 cancelled $(date '+%m-%d %H:%M')" | tee -a "$TMP/depth_crux.txt"; done_c=1
  fi
  if [ "$done_l" -eq 0 ] && grep -q '"task": "lcb", "steps": 1,' "$TMP/depth_lcb.txt" 2>/dev/null; then
    for p in $(pgrep -f 'bash coding_sweeps.sh'); do kill "$p" 2>/dev/null; done
    for p in $(pgrep -f 'bash depth_sweep.sh'); do grep -q 'TASK=lcb' "/proc/$p/environ" 2>/dev/null && kill "$p" 2>/dev/null; done
    for p in $(pgrep -f 'depth_eval.py .*--task lcb --steps 6'); do kill "$p" 2>/dev/null; done
    sleep 5
    # grade K=1 if the chain was killed before grading it
    S=/tmp/lcb_grade_stage; mkdir -p "$S"
    if ! grep -q '^K=1 GRADED' "$TMP/depth_lcb.txt" 2>/dev/null; then
      cp "$WT/lcb_grade.py" "$TMP/lcb_em120.jsonl" "$TMP/depth_lcb_k1.jsonl" "$S"/
      docker run --rm --network none --memory 4g --cpus 4 -v "$S":/w python:3.12-slim \
        python /w/lcb_grade.py /w/lcb_em120.jsonl /w/depth_lcb_k1.jsonl /w/depth_lcb_k1.graded.json 2>&1 | grep -E '^GRADED' | sed 's/^/K=1 /' | tee -a "$TMP/depth_lcb.txt"
      cp "$S/depth_lcb_k1.graded.json" "$TMP/" 2>/dev/null
    fi
    echo "lcb: K=6 cancelled; running K=3 $(date '+%m-%d %H:%M')" | tee -a "$TMP/depth_lcb.txt"
    TASK=lcb DATA=$TMP/lcb_em120.jsonl N=120 STEPS="3" BATCH=4 MAXNEW=3072 GPU=5 bash depth_sweep.sh
    cp "$WT/lcb_grade.py" "$TMP/lcb_em120.jsonl" "$TMP/depth_lcb_k3.jsonl" "$S"/
    docker run --rm --network none --memory 4g --cpus 4 -v "$S":/w python:3.12-slim \
      python /w/lcb_grade.py /w/lcb_em120.jsonl /w/depth_lcb_k3.jsonl /w/depth_lcb_k3.graded.json 2>&1 | grep -E '^GRADED' | sed 's/^/K=3 /' | tee -a "$TMP/depth_lcb.txt"
    cp "$S/depth_lcb_k3.graded.json" "$TMP/" 2>/dev/null
    echo "lcb K=3 done $(date '+%m-%d %H:%M')" | tee -a "$TMP/depth_lcb.txt"; done_l=1
  fi
  sleep 120
done
echo "DROP_EXTRAPOLATION done crux=$done_c lcb=$done_l $(date '+%m-%d %H:%M')"
