#!/bin/bash
# Item 4 data: replay each of the 11 valid eval tasks' reference solutions through
# the real terminus-2 harness with the canned oracle server (CPU only), so the
# logged trajectories carry REAL terminal output and a wait-until-prompt turn
# instead of the synthetic empty turn-2. Harvest resolved runs -> oracle_real.jsonl.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
TB=/home/jli199/terminal_bench_eval/tb_tasks
PORT=${PORT:-8160}
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
REP=$TMP/oracle_real.txt; : > "$REP"
echo "=== ORACLE_REAL start $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
cd /home/jli199/terminal_bench_eval || exit 1
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a
rids=""
for t in $ALL11; do
  rid=oracle_real_$t; rids="$rids $rid"
  "$PY" "$WT/oracle_canned_server.py" --task "$t" --port "$PORT" > "$TMP/oracle_srv_$t.log" 2>&1 &
  spid=$!
  for _ in $(seq 1 30); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "$t: canned server never came up" | tee -a "$REP"; kill "$spid" 2>/dev/null; continue; }
  rm -rf "/tmp/tb_runs/$rid"
  PYTHONPATH="$WT" timeout 2400 .venv/bin/tb run \
    --dataset-path "$TB" --agent terminus-2 --model "openai/oracle_real" \
    --agent-kwarg api_base="http://127.0.0.1:$PORT/v1" \
    -t "$t" --n-attempts 1 --n-concurrent 1 --global-agent-timeout-sec 1200 \
    --output-path /tmp/tb_runs --run-id "$rid" --no-livestream >> "$TMP/oracle_real.tb.log" 2>&1
  kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
  res=$(cat /tmp/tb_runs/$rid/*/*/results.json 2>/dev/null | grep -oE '"is_resolved": *(true|false)' | head -1)
  fm=$(cat /tmp/tb_runs/$rid/*/*/results.json 2>/dev/null | grep -oE '"failure_mode": *"[^"]*"' | head -1)
  turns=$(ls -d /tmp/tb_runs/$rid/*/*/agent-logs/episode-* 2>/dev/null | wc -l)
  echo "$t: ${res:-no results} ${fm:-} episodes=$turns server: $(grep -oE '\-> [a-z-]+' "$TMP/oracle_srv_$t.log" | tr '\n' ' ')" | tee -a "$REP"
done
"$PY" "$WT/harvest_traj.py" $rids --out "$TMP/oracle_real.jsonl" >> "$REP" 2>&1
echo "harvested rows: $(wc -l < "$TMP/oracle_real.jsonl" 2>/dev/null || echo 0) -> $TMP/oracle_real.jsonl" | tee -a "$REP"
echo "=== ORACLE_REAL done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
