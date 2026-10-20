#!/bin/bash
# Rejection-sample deepseek-v4.1-flash on the validated TB task VARIANTS to
# distill TARGET trajectories. Each round: refresh the scaffolded pool (picks up
# new valid tasks), run pass@ATT, harvest is_resolved, dedup-append. Loop to TARGET.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
VPY=/home/jli199/boptim-agent/.venv/bin/python
POOL=/home/jli199/boptim_scratch/tb_variants/pool
OUT=/home/jli199/boptim_scratch/gen_out/variant_trajectories.jsonl
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PROG=$TMP/distill_300_progress.txt
TARGET=${TARGET:-300}; ATT=${ATT:-4}; CONC=${CONC:-8}
touch "$OUT"; : > "$PROG"
cnt(){ local n; n=$(wc -l < "$OUT" 2>/dev/null); echo "${n:-0}"; }
echo "[distill] start $(date '+%m-%d %H:%M') target=$TARGET att=$ATT have=$(cnt)" >> "$PROG"
round=0
END=$(( $(date +%s) + 115200 ))   # 32h cap
while [ "$(cnt)" -lt "$TARGET" ] && [ "$(date +%s)" -lt "$END" ]; do
  round=$((round+1))
  bash "$WT/scaffold_variants.sh" >/dev/null 2>&1
  ntasks=$(ls -d "$POOL"/*/ 2>/dev/null | wc -l)
  rid="variant_distill_r${round}"; rm -rf /tmp/tb_runs/$rid
  TB_MAX_TOKENS=32000 TASKS_DIR="$POOL" RUNID="$rid" MODEL=openrouter/deepseek/deepseek-v4.1-flash ATT="$ATT" CONC="$CONC" \
    bash "$WT/gen_traj.sh" >> "$TMP/distill_gen.log" 2>&1
  "$VPY" "$WT/harvest_traj.py" "$rid" --out "$TMP/${rid}.jsonl" >/dev/null 2>&1
  # dedup-append
  "$VPY" - "$OUT" "$TMP/${rid}.jsonl" <<'PY'
import sys, json, hashlib, os
out, new = sys.argv[1], sys.argv[2]
seen=set()
if os.path.exists(out):
    for l in open(out):
        if l.strip():
            try: seen.add(hashlib.md5(json.dumps(json.loads(l).get("messages"),sort_keys=True).encode()).hexdigest())
            except: pass
added=0
if os.path.exists(new):
    with open(out,"a") as w:
        for l in open(new):
            if not l.strip(): continue
            try: o=json.loads(l)
            except: continue
            h=hashlib.md5(json.dumps(o.get("messages"),sort_keys=True).encode()).hexdigest()
            if h in seen: continue
            seen.add(h); w.write(l if l.endswith("\n") else l+"\n"); added+=1
print(added)
PY
  echo "$(date '+%m-%d %H:%M') round $round tasks=$ntasks have=$(cnt)/$TARGET" >> "$PROG"
  find /tmp/tb_runs/$rid -name "*.cast" -delete 2>/dev/null; rm -rf /tmp/tb_runs/$rid 2>/dev/null
  docker image prune -f >/dev/null 2>&1
done
echo "[distill] done $(date '+%m-%d %H:%M') have=$(cnt)" >> "$PROG"
