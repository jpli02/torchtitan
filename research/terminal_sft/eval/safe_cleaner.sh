#!/bin/bash
# Disk cleaner for /tmp/tb_runs that only touches FINISHED trials.
#
# The previous cleaners deleted every *.cast (no age filter) and stripped
# agent-logs/panes/sessions of any trial dir that had a results.json. terminal
# -bench writes results.json early and streams the cast into sessions/ while
# the agent runs, so both cleaners hit live trials: the harness then failed
# with "File '/logs/agent.cast' does not exist" and FileNotFoundError on
# agent-logs/episode-N/response.txt -> failure_mode unknown_agent_error on
# 16/24 trials of tb80sft_v2_10k (VOID). This one requires trial_ended_at to
# be set in results.json before it removes anything, and keeps results.json
# and commands.txt so the tallies still work.
#
# usage: research/terminal_sft/eval/safe_cleaner.sh <watch_pid> [interval_s]
WATCH=${1:?pid to follow}
EVERY=${2:-120}
while kill -0 "$WATCH" 2>/dev/null; do
  for r in /tmp/tb_runs/*/*/*/results.json; do
    [ -f "$r" ] || continue
    grep -q '"trial_ended_at": "20' "$r" || continue
    d=$(dirname "$r")
    rm -rf "$d/agent-logs" "$d/panes" "$d/sessions" 2>/dev/null
  done
  sleep "$EVERY"
done
echo "safe cleaner exited with pid $WATCH"; df -h /tmp | tail -1
