#!/bin/bash
# Validation of the fill-on search, second part (replaces the live phase of bo3_followup.sh):
#   1. held-out set: the three remaining search-set routers that hold the floor (#14, #13, #20), so that ALL five are reported,
#      not only the two the picker chose (#17, #12);
#   2. live set (1,350): gate 0.25 + correction, #12, #17, gate 0.3 + correction, gate 0.2 + correction, plain 0.25, plain 0.2.
# Start detached:  setsid nohup bash boptim_lcb/bo3_followup2.sh >> $TMP/bo3_followup.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
M=/home/jli199/boptim_scratch/bo_bfcl_router/fill_maps.pt
R12="0.2372,0.29355,0.4755|maps=$M"
R17="0.2358,0.20185,0.5019|maps=$M"
cd /home/jli199/boptim-agent || exit 1
# exact thresholds of #14, #13, #20 from the search record (floor-holding proposals not yet on the held-out set)
REST=$(./.venv/bin/python -c "
import json
P = '/home/jli199/boptim_scratch/bo_bfcl_router'
full = next(json.loads(l) for l in open(P + '/evaluations.jsonl') if [round(v, 3) for v in json.loads(l)['thresholds'][:3]] == [1.0, 1.0, 1.0])
done = {tuple(round(v, 4) for v in json.loads(l)['thresholds'][:3]) for l in open(P + '/lab_heldout/evaluations.jsonl') if l.strip()}
out = []
for l in open(P + '_fill/evaluations.jsonl'):
    e = json.loads(l)
    k = tuple(round(v, 4) for v in e['thresholds'][:3])
    if e.get('status') == 'ok' and e['acc_pct'] >= full['acc_pct'] - 1.7 and k not in done and len(set(k)) > 1:
        done.add(k)
        out.append(','.join(repr(round(float(v), 6)) for v in e['thresholds'][:3]) + '|maps=$M')
print(';'.join(out))
")
echo "held-out, remaining floor-holding routers: ${REST//$M/MAPS} $(date '+%m-%d %H:%M')"
if [ -n "$REST" ]; then
  SET=heldout ROUTERS="$REST" ./.venv/bin/python -u "$WT/boptim_lcb/lab_eval.py" >> "$TMP/lab_eval_heldout.log" 2>&1
  echo "held-out validation, second part, returned $? $(date '+%m-%d %H:%M')"
fi
SET=live ROUTERS="0.25,0.25,0.25|maps=$M;$R12;$R17;0.3,0.3,0.3|maps=$M;0.2,0.2,0.2|maps=$M;0.25,0.25,0.25;0.2,0.2,0.2" \
  ./.venv/bin/python -u "$WT/boptim_lcb/lab_eval.py" >> "$TMP/lab_eval_live.log" 2>&1
echo "live validation returned $? $(date '+%m-%d %H:%M')"
