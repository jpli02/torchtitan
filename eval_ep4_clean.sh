#!/bin/bash
# Clean 11-task x 2 eval (lever-1 serving, full recurrence, ungated) of the take-turns r1
# 4-true-epoch checkpoints: epoch 4 (fully fit: 99.0% token acc on new rows) then epoch 2.
# Waits for the first guard eval to finish and GPU 5 to be free. Compare: r1 @1.67 ep 9/22,
# v3+lever-1 8/22; the earlier ep1/ep2 evals (6/22 each) ran under node contention.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
REP=$TMP/eval_ep4_clean.txt; : > "$REP"
cd "$WT" || exit 1
END=$(( $(date +%s) + 172800 ))
while [ "$(date +%s)" -lt "$END" ]; do
  if grep -q 'GUARD EVAL done' "$TMP/eval_guard_easy.txt" 2>/dev/null; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 5 | tr -d ' ')
    [ "${free:-0}" -ge 40000 ] && break
  fi
  sleep 300
done
echo "=== EP4/EP2 clean eval on GPU 5 $(date '+%m-%d %H:%M') load=$(uptime | sed 's/.*load average: //') ===" | tee -a "$REP"
export OURO_SEQ_CAP=12000
agg() {
"$PY" - "$1" >> "$REP" 2>&1 <<'PYEOF'
import json, glob, sys, collections
name=sys.argv[1]; per=collections.defaultdict(lambda:[0,0]); fm=collections.Counter()
for f in glob.glob(f"/tmp/tb_runs/{name}/*/*/results.json"):
    d=json.load(open(f)); t=d.get("task_id"); per[t][1]+=1; per[t][0]+= 1 if d.get("is_resolved") else 0
    fm[str(d.get("failure_mode") or ("RESOLVED" if d.get("is_resolved") else "other"))]+=1
tot=sum(v[0] for v in per.values()); n=sum(v[1] for v in per.values())
print(f"{name}: {tot}/{n} solved  tasks>=1: {sum(1 for v in per.values() if v[0]>0)}/{len(per)}")
for t,(s,c) in sorted(per.items()): print(f"   {'SOLVED' if s else '  --  '} {t} {s}/{c}")
print("failure_modes:", dict(fm))
PYEOF
L=/home/jli199/terminal_bench_eval/logs/$1_server.log
grep -oE 'secs=[0-9.]+ tok/s=[0-9.]+' "$L" 2>/dev/null | awk -F'[= ]' '{s+=$2; t+=$4; n++} END {if(n) printf "   server: %d turns, mean %.1f s/turn, %.1f tok/s\n", n, s/n, t/n}' | tee -a "$REP"
}
for k in 4 2; do
  GPU=5 GATE=0 CKPT=/home/jli199/boptim_scratch/ouro_tvary_r1_4ep_ep${k}_clean NAME=tvary_r1_4ep_ep${k}_clean PORT=8152 ATT=2 MINFREE=40000 \
    TASKS="$ALL11" bash eval_gated.sh > "$TMP/tvary_r1_4ep_ep${k}_clean_eval.log" 2>&1
  agg tvary_r1_4ep_ep${k}_clean
done
echo "compare: r1 @1.67ep 9/22 (6 tasks) | v3+lever-1 8/22 | contended ep1/ep2 6/22" | tee -a "$REP"
echo "=== EP4/EP2 clean eval done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
