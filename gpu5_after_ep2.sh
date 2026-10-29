#!/bin/bash
# Wait for the take-turns epoch-2 eval (tb run 1509604) to finish and free GPU 5,
# aggregate its result into the item-1 report, then run the MATH-500 depth sweep on GPU 5.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
END=$(( $(date +%s) + 43200 ))
while [ "$(date +%s)" -lt "$END" ]; do
  alive=$(ps -o pid= -p 1509604 2>/dev/null | wc -l)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 | tr -d ' ')
  [ "$alive" -eq 0 ] && [ "${used:-99999}" -lt 2000 ] && break
  sleep 120
done
echo "ep2 eval finished / GPU 5 free at $(date '+%m-%d %H:%M')" | tee -a "$TMP/train_tvary_r1_4ep.txt"
"$PY" - tvary_r1_4ep_ep2 >> "$TMP/train_tvary_r1_4ep.txt" 2>&1 <<'PYEOF'
import json, glob, sys, collections
name=sys.argv[1]; per=collections.defaultdict(lambda:[0,0]); fm=collections.Counter()
for f in glob.glob(f"/tmp/tb_runs/{name}/*/*/results.json"):
    d=json.load(open(f)); t=d.get("task_id"); per[t][1]+=1; per[t][0]+= 1 if d.get("is_resolved") else 0
    fm[str(d.get("failure_mode") or ("RESOLVED" if d.get("is_resolved") else "other"))]+=1
tot=sum(v[0] for v in per.values()); n=sum(v[1] for v in per.values())
print(f"{name}: {tot}/{n} solved  tasks>=1: {sum(1 for v in per.values() if v[0]>0)}/{len(per)}")
for t,(s,c) in sorted(per.items()): print(f"   {'SOLVED' if s else '  --  '} {t} {s}/{c}")
print("failure_modes:", dict(fm))
print("(epochs 3 and 4 NOT evaluated: batch paused 10-29 13:23 on user request)")
PYEOF
cd "$WT" || exit 1
TASK=math DATA=$TMP/math500.jsonl N=150 STEPS="4 2 1 6 3 5" BATCH=4 MAXNEW=3072 GPU=5 bash depth_sweep.sh
