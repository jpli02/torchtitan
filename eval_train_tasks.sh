#!/bin/bash
# Pass@1 on the TRAINING tasks of the take-turns distillation data (the 75 variant tasks that
# produced r1's rows), 1 attempt, 900 s (the generation budget), lever-1 serving, full recurrence:
#   v3 (before continuation) | take-turns r1 (1.67 epochs, 94.7% token fit) | 4-epoch ep4 (99.0% fit)
# Tells whether token-level fit becomes task success ON-DISTRIBUTION. Two at a time (docker contention).
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
POOL=/home/jli199/boptim_scratch/tb_variants/pool
# must be a LOCAL filesystem: the harness bind-mounts each trial's log dir into the container, and
# docker's root cannot write to the NFS home (that is why the first attempt produced 0 trials)
OUT=/srv/scratch/jli199/tb_runs
mkdir -p "$OUT"
TASKS=$(cat "$TMP/train_tasks_r1.txt")
REP=$TMP/eval_train_tasks.txt; : > "$REP"
cd "$WT" || exit 1
export OURO_SEQ_CAP=12000
echo "=== TRAIN-TASK pass@1 (75 r1 training tasks x1, 900 s) $(date '+%m-%d %H:%M') load=$(uptime | sed 's/.*load average: //') ===" | tee -a "$REP"
agg() {
"$PY" - "$1" "$OUT" >> "$REP" 2>&1 <<'PYEOF'
import json, glob, sys, collections
name, out = sys.argv[1], sys.argv[2]; n_ok = n = 0; fm = collections.Counter()
for f in glob.glob(f"{out}/{name}/*/*/results.json"):
    d = json.load(open(f)); n += 1; n_ok += bool(d.get("is_resolved")); fm[str(d.get("failure_mode") or ("RESOLVED" if d.get("is_resolved") else "other"))] += 1
print(f"{name}: {n_ok}/{n} training tasks solved ({100*n_ok/max(n,1):.1f}%)  failure_modes={dict(fm)}")
PYEOF
L=/home/jli199/terminal_bench_eval/logs/$1_server.log
grep -oE 'secs=[0-9.]+ tok/s=[0-9.]+' "$L" 2>/dev/null | awk -F'[= ]' '{s+=$2; t+=$4; n++} END {if(n) printf "   server: %d turns, %.1f s/turn, %.1f tok/s\n", n, s/n, t/n}' | tee -a "$REP"
}
one() {  # one <gpu> <name> <ckpt>
  GPU=$1 GATE=0 CKPT=$3 NAME=$2 PORT=$((8160+$1)) ATT=1 MINFREE=40000 DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=900 TASKS="$TASKS" \
    bash eval_gated.sh > "$TMP/$2.log" 2>&1
  agg "$2"
}
( one 4 train75_v3 /home/jli199/boptim_scratch/ouro_tb80sft26b_v3_clean ) &
( one 5 train75_ep4 /home/jli199/boptim_scratch/ouro_tvary_r1_4ep_ep4_clean ) &
wait
gpu=""; while [ -z "$gpu" ]; do gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | awk -F, '($1==4||$1==5) && $2+0>=40000 {print $1; exit}'); [ -n "$gpu" ] || sleep 120; done
one "$gpu" train75_r1 /home/jli199/boptim_scratch/ouro_tvary_r1_clean
echo "reference: generation-time student-only solve rate 2.6%; teacher-corrected rows exist for all 75 tasks" | tee -a "$REP"
echo "=== TRAIN-TASK pass@1 done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
