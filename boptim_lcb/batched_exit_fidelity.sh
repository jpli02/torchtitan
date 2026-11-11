#!/bin/bash
# Does batched generation under TRUE decode-time early exit reproduce the batch-1 outputs? If yes, the BO's accuracy generation can
# run at batch 8 (about 5x cheaper) while the speed calibration stays at batch 1.
#   GPU=0 N=24 bash boptim_lcb/batched_exit_fidelity.sh      -> prints exact-match / pass-agreement against the batch-1 run
set -u
GPU=${GPU:-0}; N=${N:-24}; BATCH=${BATCH:-8}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
DATA=$TMP/lcb_all.jsonl
REF=$(ls -d /home/jli199/boptim_scratch/bo_lcb_router_pooled/q0.367-0.367-0.367_*/ | head -n 1)gen.jsonl
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$DATA" --task lcb --steps 4 \
  --n "$N" --batch "$BATCH" --max_new 1024 --assistant_prefix '```python\n' --cfg 'decode_early_exit_threshold=[0.36666666666666664,0.36666666666666664,0.36666666666666664,1.0]' \
  --out "$TMP/fidelity_b${BATCH}.jsonl" > "$TMP/fidelity_b${BATCH}.log" 2>&1
grep '^RESULT' "$TMP/fidelity_b${BATCH}.log" | cut -c1-200
"$PY" - "$REF" "$TMP/fidelity_b${BATCH}.jsonl" <<'PYEOF'
import json, sys
ref = {r["i"]: r for r in (json.loads(l) for l in open(sys.argv[1]) if l.strip()) if "i" in r}
new = [r for r in (json.loads(l) for l in open(sys.argv[2]) if l.strip()) if "i" in r]
same = sum(1 for r in new if ref[r["i"]]["gen"] == r["gen"])
same_pred = sum(1 for r in new if (ref[r["i"]]["pred"] or "") == (r["pred"] or ""))
dn = [abs(ref[r["i"]]["ntok"] - r["ntok"]) for r in new]
print(f"n={len(new)}: identical generations {same}/{len(new)}, identical extracted code {same_pred}/{len(new)}, mean |token-count diff| {sum(dn)/len(dn):.1f}, max {max(dn)}")
for r in new:
    if ref[r["i"]]["gen"] != r["gen"]:
        a, b = ref[r["i"]]["gen"], r["gen"]
        k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        print(f"  problem {r['i']}: diverges at char {k} of {len(a)} / {len(b)}")
PYEOF
