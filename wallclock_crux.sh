#!/bin/bash
# Real wall-clock of the routers under the TRUE decode-time early exit (batch 1, the serving
# path): CRUXEval items 0-99, greedy, 2,048-token budget. Reports accuracy and tokens/s.
#   full depth | released gate q=0.3 (true exit) | per-step (0.1,0.5,0.5) (true exit)
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
GPU=${GPU:-2}; N=${N:-100}
REP=$TMP/wallclock_crux.txt; : > "$REP"
cd "$WT" || exit 1
echo "=== WALL-CLOCK (true early exit, batch 1) CRUX n=$N gpu $GPU $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
run() {  # run <label> <extra args...>
  local label=$1; shift
  echo "--- $label start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$TMP/cruxeval.jsonl" --task crux --steps 4 \
    --n "$N" --batch 1 --max_new 2048 "$@" --out "$TMP/wallclock_crux_${label}.jsonl" > "$TMP/wallclock_crux_${label}.log" 2>&1
  grep '^RESULT' "$TMP/wallclock_crux_${label}.log" | sed "s/^RESULT /RESULT $label /" | tee -a "$REP"
  grep -q '^RESULT' "$TMP/wallclock_crux_${label}.log" || echo "$label FAILED: $(grep -E 'Error' "$TMP/wallclock_crux_${label}.log" | tail -1 | cut -c1-160)" | tee -a "$REP"
}
run full
run gate0.3 --cfg decode_early_exit_threshold=0.3
run steps0.1-0.5-0.5 --cfg "decode_early_exit_threshold=[0.1,0.5,0.5,1.0]"
"$PY" - "$REP" <<'PYEOF'
import json, sys
rows = []
for l in open(sys.argv[1]):
    if l.startswith("RESULT "):
        lab = l.split()[1]; r = json.loads(l[l.index("{"):]); rows.append((lab, r))
print(f"{'router':>18} {'acc':>6} {'tok/s':>6} {'s/item':>7} {'tokens':>7} {'trunc':>6}")
for lab, r in rows:
    print(f"{lab:>18} {r['acc']:6.3f} {r['tok_s']:6.2f} {r['wall_s']/r['n']:7.1f} {r['mean_tokens']:7.0f} {r['trunc_rate']:6.2f}")
PYEOF
echo "=== WALL-CLOCK done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
