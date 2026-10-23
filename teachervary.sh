#!/bin/bash
# Teacher/student TAKE-TURNS with VARIED patterns (distillation, no RL). Each free
# GPU runs a shard of the 139 training tasks under a DIFFERENT teacher-turns
# pattern, so harvested data carries teacher corrections at different turns
# (student's own states throughout). pass@2, harvest union -> teachervary_round1.jsonl.
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
POOL=/home/jli199/boptim_scratch/tb_variants/pool
PY=/home/jli199/torchtitan/.venv/bin/python
ATT=${ATT:-2}
PATTERNS=(${PATTERNS:-odd every:3 every:2 prob:0.5 after:6 odd})
OUT=$TMP/teachervary_round1.jsonl
REP=$TMP/teachervary_round1.txt; : > "$REP"
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | awk -F, '$2+0>=44000{print $1}')
NG=${#GPUS[@]}
[ "$NG" -ge 1 ] || { echo "ABORT no free GPU" | tee -a "$REP"; exit 1; }
mapfile -t TASKS < <(ls -d "$POOL"/*/ | xargs -n1 basename | sort)
echo "=== TEACHERVARY r1 start $(date '+%m-%d %H:%M') tasks=${#TASKS[@]} gpus=(${GPUS[*]}) patterns=(${PATTERNS[*]:0:$NG}) att=$ATT ===" | tee -a "$REP"
declare -a SH
for i in "${!TASKS[@]}"; do k=$(( i % NG )); SH[$k]="${SH[$k]:-} ${TASKS[$i]}"; done
pids=()
for k in $(seq 0 $((NG-1))); do
  g=${GPUS[$k]}; port=$((8410+k)); patt=${PATTERNS[$k]}; rid="tvary_r1_sh${k}"; sj="$TMP/${rid}.jsonl"
  echo "  shard $k gpu $g port $port pattern=$patt tasks=$(echo ${SH[$k]}|wc -w)" | tee -a "$REP"
  GPU=$g PORT=$port RUNID="$rid" SHARD_TASKS="${SH[$k]}" OUTJSONL="$sj" ATT="$ATT" TSPEC="$patt" \
    bash "$TMP/teachervary_shard.sh" >> "$TMP/${rid}.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
: > "$OUT"
"$PY" - "$OUT" $(for k in $(seq 0 $((NG-1))); do echo "$TMP/tvary_r1_sh${k}.jsonl"; done) <<'PYEOF'
import sys, json, hashlib, os
out=sys.argv[1]; seen=set(); n=0
with open(out,"w") as w:
    for f in sys.argv[2:]:
        if not os.path.exists(f): continue
        for l in open(f):
            if not l.strip(): continue
            try: o=json.loads(l)
            except: continue
            h=hashlib.md5(json.dumps(o.get("messages"),sort_keys=True).encode()).hexdigest()
            if h in seen: continue
            seen.add(h); w.write(l if l.endswith("\n") else l+"\n"); n+=1
print("harvested rows:", n)
PYEOF
echo "=== TEACHERVARY r1 done $(date '+%m-%d %H:%M') rows=$(wc -l < "$OUT" 2>/dev/null || echo 0) -> $OUT ===" | tee -a "$REP"
