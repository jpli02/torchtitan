#!/bin/bash
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
echo "=== now $(date '+%m-%d %H:%M') (relaunched 10-21 12:50) ==="
echo "--- GEN ---"
grep -E "TEACHERVARY r1 (start|done)|harvested rows" "$TMP/teachervary_round1.txt" 2>/dev/null
echo "gen waiter: $(pgrep -f '[t]eachervary.sh' | wc -l)  servers: $(pgrep -af '[o]uro_openai_server' | grep -c tvary)  final rows: $(wc -l < $TMP/teachervary_round1.jsonl 2>/dev/null || echo 0)"
$PY - <<'PYEOF'
import json, glob, collections
res=collections.Counter(); solved=collections.Counter(); per_shard=collections.Counter()
for f in glob.glob("/tmp/tb_runs/tvary_r1_sh*/*/*/results.json"):
    try: d=json.load(open(f))
    except: continue
    sh=f.split("/tmp/tb_runs/")[1].split("/")[0]
    r=d.get("is_resolved"); res["r" if r else "f"]+=1; per_shard[sh]+=1
    if r: solved[d.get("task_id")]+=1
t=sum(res.values())
print(f"graded {t}/278  resolved {res['r']} ({100*res['r']/max(t,1):.0f}%)  distinct solved tasks {len(solved)}")
print("per-shard graded:", dict(sorted(per_shard.items())))
PYEOF
tt=0; tc=0
for k in 0 1 2 3 4 5; do
  L=/home/jli199/terminal_bench_eval/logs/tvary_r1_sh${k}_server.log
  a=$(grep -c 'TEACHER turn' "$L" 2>/dev/null); b=$(grep -c 'teacher call failed' "$L" 2>/dev/null)
  tt=$((tt+a)); tc=$((tc+b))
done
echo "TEACHER turns fired: $tt   teacher-call failures: $tc"
echo "--- TRAIN (wandb, staged) ---"
cat "$TMP/train_tvary_r1.txt" 2>/dev/null | head -14
echo "train waiter: $(pgrep -f '[t]rain_tvary.sh' | wc -l)"
echo "--- disk/GPU ---"
df -h /srv 2>/dev/null | tail -1 | awk '{print $5" used "$4" free"}'
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | awk -F, '{printf "g%s:%s%% ",$1,$2}'; echo
