#!/bin/bash
# Round-2 clean 6-pattern take-turns generation (new OpenRouter key, full teacher
# coverage). Waits until round-1's WHOLE chain (gen->train->eval) is done so its
# 6 tb harnesses never overlap round-1's eval harness (node contention lesson).
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
END=$(( $(date +%s) + 172800 ))
while [ "$(date +%s)" -lt "$END" ]; do
  grep -qE "TRAIN_TVARY r1 done|^ABORT" "$TMP/train_tvary_r1.txt" 2>/dev/null && break
  sleep 300
done
sleep 60
pkill -f "[o]uro_openai_server" 2>/dev/null; sleep 5
bash /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/teachervary_r2.sh
