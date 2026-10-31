#!/bin/bash
# Released-gate operating curve (stock gather-mode selection: all 4 steps computed, the step
# where the cumulative exit probability crosses q is used; no speedup, but accuracy and
# mean selected loops are exact for the gate's decisions).
#   TASK=crux DATA=... N=200 THRS="0.9 0.7 0.5 0.3 0.2 0.1" GPU=2 bash gate_sweep.sh
set -u
: "${TASK:?}"; : "${DATA:?}"; : "${GPU:?}"
N=${N:-200}; THRS=${THRS:-"0.9 0.7 0.5 0.3 0.2 0.1"}; BATCH=${BATCH:-6}; MAXNEW=${MAXNEW:-2048}
OFFSET=${OFFSET:-0}   # skip this many items first (held-out validation); "none" in THRS = full depth, no threshold
CKPT=${CKPT:-/home/jli199/boptim_scratch/ouro_base_kvshare}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
REP=$TMP/gate_${TASK}.txt
cd "$WT" || exit 1
echo "=== GATE SWEEP $TASK n=$N thr=($THRS) batch=$BATCH max_new=$MAXNEW gpu=$GPU $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
for Q in $THRS; do
  echo "--- q=$Q start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  case "$Q" in none) TARG="" ;; *,*) TARG="--threshold_steps $Q" ;; *) TARG="--threshold $Q" ;; esac   # "q1,q2,q3" = per-step thresholds
  SUF="q${Q}"; [ "$OFFSET" != "0" ] && SUF="q${Q}_o${OFFSET}"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" \
    --task "$TASK" --steps 4 $TARG --n "$N" --batch "$BATCH" --max_new "$MAXNEW" --seed_offset "$OFFSET" \
    --out "$TMP/gate_${TASK}_${SUF}.jsonl" > "$TMP/gate_${TASK}_${SUF}.log" 2>&1
  grep '^RESULT' "$TMP/gate_${TASK}_${SUF}.log" | sed "s/^RESULT /RESULT offset=$OFFSET /" | tee -a "$REP"
  grep -q '^RESULT' "$TMP/gate_${TASK}_${SUF}.log" || echo "q=$Q offset=$OFFSET FAILED: $(grep -E 'Error' "$TMP/gate_${TASK}_${SUF}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"
done
echo "=== GATE SWEEP $TASK done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
"$PY" - "$REP" <<'PYEOF'
import json, sys
rows = []
for l in open(sys.argv[1]):
    if l.startswith("RESULT "):
        r = json.loads(l[l.index("{"):]); r["offset"] = l.split("offset=")[1].split()[0] if "offset=" in l else "0"; rows.append(r)
print(f"{'offset':>6} {'threshold':>16} {'acc':>6} {'loops':>6} {'parse':>6} {'trunc':>6} {'tokens':>7}")
for r in sorted(rows, key=lambda r: (r['offset'], -r.get('mean_loops', 4))):
    t = r['threshold']; t = ",".join(f"{x:g}" for x in t) if isinstance(t, list) else f"{t}"
    print(f"{r['offset']:>6} {t:>16} {r['acc']:6.3f} {r.get('mean_loops', 4):6.2f} {r['parse_rate']:6.3f} {r['trunc_rate']:6.3f} {r['mean_tokens']:7.0f}")
PYEOF
