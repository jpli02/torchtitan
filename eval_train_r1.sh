#!/bin/bash
# Train-task pass@1 for the take-turns r1 checkpoint on GPU 2 now (instead of waiting for GPUs 4/5),
# same protocol as eval_train_tasks.sh. Stops the chain shell first so it does not re-run r1 later.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
POOL=/home/jli199/boptim_scratch/tb_variants/pool
OUT=/srv/scratch/jli199/tb_runs
TASKS=$(cat "$TMP/train_tasks_r1.txt")
REP=$TMP/eval_train_tasks.txt
cd "$WT" || exit 1
for p in $(pgrep -f 'bash eval_train_tasks.sh'); do kill "$p" 2>/dev/null && echo "stopped chain shell $p (v3/ep4 subshells keep running)"; done
export OURO_SEQ_CAP=12000
GPU=2 GATE=0 CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_clean NAME=train75_r1 PORT=8162 ATT=1 MINFREE=40000 DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=900 TASKS="$TASKS" \
  bash eval_gated.sh > "$TMP/train75_r1.log" 2>&1
"$PY" - train75_r1 "$OUT" >> "$REP" 2>&1 <<'PYEOF'
import json, glob, sys, collections
name, out = sys.argv[1], sys.argv[2]; n_ok = n = 0; fm = collections.Counter()
for f in glob.glob(f"{out}/{name}/*/*/results.json"):
    d = json.load(open(f)); n += 1; n_ok += bool(d.get("is_resolved")); fm[str(d.get("failure_mode") or ("RESOLVED" if d.get("is_resolved") else "other"))] += 1
print(f"{name}: {n_ok}/{n} training tasks solved ({100*n_ok/max(n,1):.1f}%)  failure_modes={dict(fm)}")
PYEOF
grep -oE 'secs=[0-9.]+ tok/s=[0-9.]+' /home/jli199/terminal_bench_eval/logs/train75_r1_server.log 2>/dev/null | awk -F'[= ]' '{s+=$2; t+=$4; n++} END {if(n) printf "   server: %d turns, %.1f s/turn, %.1f tok/s\n", n, s/n, t/n}' | tee -a "$REP"
echo "train75_r1 done $(date '+%m-%d %H:%M')" | tee -a "$REP"
