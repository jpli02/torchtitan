#!/bin/bash
# r1+r2 COMBINED take-turns training: wait for round-2 generation to finish,
# union+dedup round-1 (112 rows) + round-2 trajectories -> teachervary_r12.jsonl,
# then run train_tvary_r12.sh (continue v3 weights-only + replay, wandb offline,
# export, 12-task eval vs 8/22 and vs r1's 9/22).
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
END=$(( $(date +%s) + 259200 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -q "TEACHERVARY r2 done" "$TMP/teachervary_round2.txt" 2>/dev/null && break
  sleep 300
done
sleep 30
"$PY" - "$TMP/teachervary_r12.jsonl" "$TMP/teachervary_round1.jsonl" "$TMP/teachervary_round2.jsonl" <<'PYEOF'
import sys, json, hashlib, os
out = sys.argv[1]; seen = set(); n = 0; per = {}
with open(out, "w") as w:
    for f in sys.argv[2:]:
        if not os.path.exists(f): continue
        c = 0
        for l in open(f):
            if not l.strip(): continue
            try: o = json.loads(l)
            except Exception: continue
            h = hashlib.md5(json.dumps(o.get("messages"), sort_keys=True).encode()).hexdigest()
            if h in seen: continue
            seen.add(h); w.write(l if l.endswith("\n") else l + "\n"); n += 1; c += 1
        per[os.path.basename(f)] = c
print("combined r1+r2 unique rows:", n, per)
PYEOF
bash "$TMP/train_tvary_r12.sh"
