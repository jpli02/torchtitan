#!/bin/bash
# /tmp filled again (the ep2 aggregation failed with ENOSPC). Aggregate ep2 from its results,
# then archive COMPLETED run dirs to boptim_scratch, keeping the runs still in progress
# (easy12_*_clean, and anything a live tb run owns).
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
A=/home/jli199/boptim_scratch/tb_runs_archive_1101
echo "before: $(df -h /tmp | awk 'NR>1{print $4" free of "$2}')"
du -sh /tmp/tb_runs /tmp/lcb_grade_stage /tmp/claude-1274873 2>/dev/null | awk '{print "  "$1"  "$2}'
"$PY" - tvary_r1_4ep_ep2_clean >> "$TMP/eval_ep4_clean.txt" 2>&1 <<'PYEOF'
import json, glob, sys, collections
name=sys.argv[1]; per=collections.defaultdict(lambda:[0,0]); fm=collections.Counter()
for f in glob.glob(f"/tmp/tb_runs/{name}/*/*/results.json"):
    d=json.load(open(f)); t=d.get("task_id"); per[t][1]+=1; per[t][0]+= 1 if d.get("is_resolved") else 0
    fm[str(d.get("failure_mode") or ("RESOLVED" if d.get("is_resolved") else "other"))]+=1
tot=sum(v[0] for v in per.values()); n=sum(v[1] for v in per.values())
print(f"{name}: {tot}/{n} solved  tasks>=1: {sum(1 for v in per.values() if v[0]>0)}/{len(per)}  (aggregated after ENOSPC)")
for t,(s,c) in sorted(per.items()): print(f"   {'SOLVED' if s else '  --  '} {t} {s}/{c}")
print("failure_modes:", dict(fm))
PYEOF
grep -A14 'ep2_clean:' "$TMP/eval_ep4_clean.txt" | head -15
mkdir -p "$A"
live=$(pgrep -af 'tb run' | grep -oE 'run-id [A-Za-z0-9_.-]+' | awk '{print $2}' | sort -u | tr '\n' ' ')
echo "live run ids: ${live:-none}"
for d in /tmp/tb_runs/*/; do
  n=$(basename "$d")
  case " $live " in *" $n "*) echo "  keep (live): $n"; continue;; esac
  mv "$d" "$A/" && echo "  archived: $n"
done
rm -rf /tmp/lcb_grade_stage
echo "after: $(df -h /tmp | awk 'NR>1{print $4" free of "$2}')"
