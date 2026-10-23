#!/bin/bash
# Wait for the greedy (temp 0) oracle12 re-eval to finish, then aggregate.
# Answers "why is the oracle not 100%": greedy ~11/11 => the 4/24 misses were
# sampling noise; greedy still missing => structural (synthetic turn-2 prompt
# mismatch / execution), not sampling.
END=$(( $(date +%s) + 28800 ))
sleep 600
while [ "$(date +%s)" -lt "$END" ]; do
  alive=$(pgrep -f 'run-id oracle12_greedy' | grep -v pgrep | grep -c .)
  n=$(ls /tmp/tb_runs/oracle12_greedy/*/*/results.json 2>/dev/null | wc -l)
  if [ "$alive" -eq 0 ] && [ "$n" -ge 1 ]; then break; fi
  sleep 180
done
echo "=== ORACLE12 GREEDY RE-EVAL (temp 0, 1 attempt, 11 valid tasks) $(date '+%m-%d %H:%M') ==="
grep -m1 "force_temperature\|exit_threshold" /home/jli199/terminal_bench_eval/logs/oracle12_greedy_server.log 2>/dev/null | cut -c1-120
/home/jli199/boptim-agent/.venv/bin/python - <<'PY'
import json, glob, collections
per={}; fm=collections.Counter()
for f in glob.glob("/tmp/tb_runs/oracle12_greedy/*/*/results.json"):
    d=json.load(open(f)); t=d.get("task_id"); r=bool(d.get("is_resolved"))
    per[t]=("RESOLVED" if r else str(d.get("failure_mode"))); fm["RESOLVED" if r else str(d.get("failure_mode"))]+=1
print(f"greedy oracle: {sum(1 for v in per.values() if v=='RESOLVED')}/{len(per)} solved")
for t,v in sorted(per.items()): print(f"   {v:<16} {t}")
print("failure_modes:", dict(fm))
print("reference: sampled (temp 0.7, pass@2) oracle12 = 20/24 (83%); 11-task equivalent ~9/11")
PY
echo "server 500s: $(grep -c '500 Internal' /home/jli199/terminal_bench_eval/logs/oracle12_greedy_server.log 2>/dev/null)"
