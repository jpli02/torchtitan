#!/bin/bash
# One-shot ~4 min after launch: did the greedy oracle server come up (temp 0.0)
# and did the tb run start? Catches a VOID immediately instead of at the
# monitor's 8h timeout.
END=$(( $(date +%s) + 240 )); while [ "$(date +%s)" -lt "$END" ]; do sleep 30; done
echo "=== greedy oracle early check $(date '+%m-%d %H:%M') ==="
curl -sf -m3 http://127.0.0.1:8150/health >/dev/null 2>&1 && echo "server 8150: UP" || echo "server 8150: DOWN"
echo "server started with: $(grep -m1 -oE 'force_temperature[^ ]*|exit_threshold=[^ ]*' /home/jli199/terminal_bench_eval/logs/oracle12_greedy_server.log 2>/dev/null | tr '\n' ' ')"
echo "VOID/ABORT in launcher log: $(grep -cE 'VOID|ABORT' /home/jli199/.claude/jobs/c0d2da0a/tmp/oracle12_greedy.log 2>/dev/null)"
echo "tb-run alive: $(pgrep -f 'run-id oracle12_greedy' | grep -v pgrep | grep -c .)   results so far: $(ls /tmp/tb_runs/oracle12_greedy/*/*/results.json 2>/dev/null | wc -l)/11"
echo "GPU3: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i 3)"
