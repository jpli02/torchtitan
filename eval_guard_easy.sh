#!/bin/bash
# TB-easy tier (the 12 tasks tagged easy) x 2 attempts, early-exit server (thr 0.2) on v3:
#   A) stock terminus-2            B) TerminusGuard (terminal-state guard + verify gate)
# Paired on the same tasks/server. Reference: 4/12 easy tasks solved reliably before.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
EASY12="hello-world fix-permissions csv-to-parquet processing-pipeline count-dataset-tokens create-bucket extract-safely git-workflow-hack grid-pattern-transform modernize-fortran-build security-vulhub-minio simple-web-scraper"
RUNTAG=${RUNTAG:-}   # optional suffix so re-runs do not overwrite earlier run dirs/reports
SFX=${RUNTAG:+_$RUNTAG}
REP=$TMP/eval_guard_easy${SFX}.txt; : > "$REP"
cd "$WT" || exit 1
# GPU=<idx> pins a card; otherwise wait for the KV-share accuracy check, then poll for a free card
gpu=${GPU:-}
if [ -z "$gpu" ]; then
  while ! grep -q 'KVSHARE ACC done' "$TMP/kvshare_acc.txt" 2>/dev/null; do sleep 120; done
  END=$(( $(date +%s) + 172800 ))
  while [ "$(date +%s)" -lt "$END" ]; do
    gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | awk -F, '$1!=5 && $2+0>=40000 {print $2+0, $1}' | sort -rn | head -1 | awk '{print $2}')
    [ -n "$gpu" ] && break; sleep 120
  done
fi
[ -n "$gpu" ] || { echo "ABORT no GPU" | tee -a "$REP"; exit 1; }
echo "=== GUARD EVAL (TB-easy 12 x2, early-exit thr 0.2, v3) gpu $gpu $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
export OURO_SEQ_CAP=12000
agg() {
"$PY" - "$1" >> "$REP" 2>&1 <<'PYEOF'
import json, glob, sys, collections, os
name=sys.argv[1]; per=collections.defaultdict(lambda:[0,0]); fm=collections.Counter(); guard=collections.Counter()
for f in glob.glob(f"/tmp/tb_runs/{name}/*/*/results.json"):
    d=json.load(open(f)); t=d.get("task_id"); per[t][1]+=1; per[t][0]+= 1 if d.get("is_resolved") else 0
    fm[str(d.get("failure_mode") or ("RESOLVED" if d.get("is_resolved") else "other"))]+=1
    g=os.path.join(os.path.dirname(f),"agent-logs","guard_stats.txt")
    if os.path.exists(g):
        try:
            for k,v in eval(open(g).read()).items(): guard[k]+=v
        except Exception: pass
tot=sum(v[0] for v in per.values()); n=sum(v[1] for v in per.values())
print(f"{name}: {tot}/{n} solved  tasks>=1: {sum(1 for v in per.values() if v[0]>0)}/{len(per)}")
for t,(s,c) in sorted(per.items()): print(f"   {'SOLVED' if s else '  --  '} {t} {s}/{c}")
print("failure_modes:", dict(fm))
if guard: print("guard interventions:", dict(guard))
PYEOF
}
GPU=$gpu GATE=0 THR=0.2 CKPT=/home/jli199/boptim_scratch/ouro_v3_earlyexit NAME=easy12_stock$SFX PORT=8151 ATT=2 MINFREE=40000 TASKS="$EASY12" bash eval_gated.sh > "$TMP/easy12_stock$SFX.log" 2>&1
agg easy12_stock$SFX
GPU=$gpu AGENT_IMPORT=guard_agent:TerminusGuard THR=0.2 CKPT=/home/jli199/boptim_scratch/ouro_v3_earlyexit NAME=easy12_guard$SFX PORT=8151 ATT=2 MINFREE=40000 TASKS="$EASY12" bash eval_gated.sh > "$TMP/easy12_guard$SFX.log" 2>&1
agg easy12_guard$SFX
echo "=== GUARD EVAL done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
