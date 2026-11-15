#!/bin/bash
# The BFCL card runs one workload at a time: live-set validation OR exit-lab experiments (together they ran out of memory once:
# live prompts need about 11 GB per validation shard).
#   1. wait for the validation's current live-set evaluation (gate 0.3) to be recorded;
#   2. pause the validation (restartable; finished points are cached);
#   3. run the exit-lab round given in ROUTERS alone on the card;
#   4. unless RESUME=0, start the validation again.
# Start detached:  ROUTERS="..." setsid nohup bash boptim_lcb/lab_takeover.sh > $TMP/lab_takeover.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
VLOG=$TMP/bfcl_validate.log
PAT="thr=\['0.300', '0.300', '0.300'\] acc="
n0=${N0:-$(grep -c "$PAT" "$VLOG")}
if pgrep -u "$USER" -f bfcl_validate.py > /dev/null; then
  until [ "$(grep -c "$PAT" "$VLOG")" -gt "$n0" ]; do
    pgrep -u "$USER" -f bfcl_validate.py > /dev/null || break
    sleep 5
  done
  echo "validation paused $(date '+%m-%d %H:%M') after: $(grep "$PAT" "$VLOG" | tail -n 1 | cut -c1-120)"
  python3 -I "$WT/kill_by_pattern.py" bfcl_round2_chain.sh launch_bfcl_validate.sh bfcl_validate.py bfcl_stage/bfcl_live_all.jsonl
fi
cd "$WT" || exit 1
WAIT=0 /home/jli199/boptim-agent/.venv/bin/python -u boptim_lcb/exit_lab_run.py >> "$TMP/${LABLOG:-exit_lab_run2.log}" 2>&1
echo "lab round returned $? $(date '+%m-%d %H:%M')"
if [ "${RESUME:-1}" = "1" ]; then
  echo "validation resumes $(date '+%m-%d %H:%M')"
  TOP=${TOP:-2} bash "$WT/boptim_lcb/launch_bfcl_validate.sh"
fi
