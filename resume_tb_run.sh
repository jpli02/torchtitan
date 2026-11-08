#!/bin/bash
# Continue a Terminal-Bench evaluation whose harness run ended (12/18-hour cap, crash): re-run only the tasks of TASKLIST
# that have no results.json under OUTROOT/NAME yet, into the same NAME (tb_progress.py sums all run sub-dirs).
#   NAME=train75_ep4_1800_s1 CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_4ep_ep4_clean TASKLIST="task1 task2 ..." GPU=1 PORT=8195 bash resume_tb_run.sh
set -u
: "${NAME:?}"; : "${CKPT:?}"; : "${TASKLIST:?}"; : "${GPU:?}"
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
POOL=${DATASET:-/home/jli199/boptim_scratch/tb_variants/pool}
OUT=${OUTROOT:-/srv/scratch/jli199/tb_runs}
cd "$WT" || exit 1
REMAINING=$(python3 - "$OUT" "$NAME" $TASKLIST <<'PYEOF'
import glob, json, os, sys
out, name, tasks = sys.argv[1], sys.argv[2], sys.argv[3:]
done = set()
for f in glob.glob(f"{out}/{name}/*/*/results.json"):
    try:
        done.add(json.load(open(f))["task_id"])
    except Exception:
        pass
print(" ".join(t for t in tasks if t not in done))
PYEOF
)
if [ -z "$REMAINING" ]; then echo "$NAME: nothing left to run"; exit 0; fi
echo "$NAME: resuming $(echo $REMAINING | wc -w) tasks on GPU $GPU $(date '+%m-%d %H:%M')" | tee -a "$TMP/eval_train_tasks.txt"
export OURO_SEQ_CAP=12000
GPU=$GPU GATE=${GATE:-0} MAXTOK=${MAXTOK:-2048} TBTIMEOUT=${TBTIMEOUT:-64800} CKPT=$CKPT NAME=$NAME PORT=${PORT:-8195} ATT=${ATT:-1} MINFREE=40000 DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=${AGENT_TIMEOUT:-1800} TASKS="$REMAINING" \
  bash eval_gated.sh > "$TMP/${NAME}_resume.log" 2>&1
python3 -I tb_progress.py "$OUT" "$NAME" | tee -a "$TMP/eval_train_tasks.txt"
