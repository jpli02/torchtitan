#!/bin/bash
# Rejection-sampling trajectory generation. Run a STRONG model (real API, as the
# terminus-2 agent -- no GPU) on converted Nemotron tasks; tb grades each trial
# with the task's own pytest suite; harvest_traj.py keeps only is_resolved
# trials and emits them as terminus-2 SFT rows. The agent's messages are already
# the exact JSON our eval agent parses, so no rendering is needed.
#
# Env: TASKS_DIR  MODEL(=openai/gpt-5-mini)  ATT(=2)  CONC(=2)  RUNID
set -u
cd /home/jli199/terminal_bench_eval || exit 1
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a
: "${TASKS_DIR:?}"; : "${RUNID:?}"
MODEL=${MODEL:-openai/gpt-5-mini}
ATT=${ATT:-2}; CONC=${CONC:-2}
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
OUT=$TMP/${RUNID}.txt; : > "$OUT"
echo "[gen] model=$MODEL tasks=$TASKS_DIR attempts=$ATT conc=$CONC $(date '+%m-%d %H:%M')" | tee -a "$OUT"

rm -rf /tmp/tb_runs/$RUNID
timeout 36000 .venv/bin/tb run \
  --dataset-path "$TASKS_DIR" \
  --agent terminus-2 --model "$MODEL" \
  --n-attempts "$ATT" --n-concurrent "$CONC" \
  --global-agent-timeout-sec 600 \
  --output-path /tmp/tb_runs --run-id "$RUNID" --no-livestream \
  >> /home/jli199/terminal_bench_eval/logs/${RUNID}.tb.log 2>&1

res=$(ls /tmp/tb_runs/$RUNID/*/*/results.json 2>/dev/null | wc -l)
solved=$(/home/jli199/torchtitan/.venv/bin/python -c "
import json,glob
n=0
for f in glob.glob('/tmp/tb_runs/$RUNID/*/*/results.json'):
    try:
        if json.load(open(f)).get('is_resolved'): n+=1
    except Exception: pass
print(n)")
echo "trials=$res resolved=$solved" | tee -a "$OUT"
echo "--- gen done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
