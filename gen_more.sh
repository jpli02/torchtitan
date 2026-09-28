#!/bin/bash
# Accumulating rejection-sampling generation to grow the deepseek trajectory
# corpus toward TARGET rows. Pulls tasks disjoint from the first batch via
# convert_harbor --skip, harvests each category to its own more_*.jsonl, then
# merges+dedups everything into deepseek_generated.jsonl after each category.
# Stops when DEEPSEEK >= TARGET or all categories are exhausted.
#
# Env: CATS N SKIP ATT CONC TARGET MODEL
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
GENTASKS=/home/jli199/boptim_scratch/gen_tasks
OUTDIR=/home/jli199/boptim_scratch/gen_out
PY=/home/jli199/torchtitan/.venv/bin/python
export TB_MAX_TOKENS="${TB_MAX_TOKENS:-8192}" AGENT_TIMEOUT="${AGENT_TIMEOUT:-900}"
CATS=${CATS:-"file_operations data_science scientific_computing security debugging data_processing"}
N=${N:-70}; ATT=${ATT:-4}; CONC=${CONC:-6}; SKIP=${SKIP:-10}; TARGET=${TARGET:-100}
MODEL=${MODEL:-openrouter/deepseek/deepseek-chat-v3-0324}
STATE=$TMP/gen_more_state.txt; : > "$STATE"
cd "$WT" || exit 1

echo "[gen_more] cats=[$CATS] N=$N skip=$SKIP att=$ATT conc=$CONC target=$TARGET model=$MODEL $(date '+%m-%d %H:%M')" | tee -a "$STATE"
for cat in $CATS; do
  echo "=== $cat: convert $N (skip $SKIP) $(date '+%H:%M') ===" | tee -a "$STATE"
  rm -rf "$GENTASKS/$cat"
  $PY convert_harbor.py --category "$cat" --n "$N" --skip "$SKIP" --out "$GENTASKS" >> "$STATE" 2>&1
  rid="genmore_${cat}_s${SKIP}"
  rm -rf /tmp/tb_runs/$rid
  TASKS_DIR="$GENTASKS/$cat" RUNID="$rid" MODEL="$MODEL" ATT="$ATT" CONC="$CONC" \
    bash gen_traj.sh >> "$STATE" 2>&1
  $PY harvest_traj.py "$rid" --out "$OUTDIR/more_${cat}_s${SKIP}.jsonl" >> "$STATE" 2>&1
  # disk hygiene
  find /tmp/tb_runs/$rid -name "*.cast" -delete 2>/dev/null
  rm -rf /tmp/tb_runs/$rid 2>/dev/null
  docker image prune -f >/dev/null 2>&1
  cnt=$($PY merge_deepseek.py)
  echo "  $cat done -> $cnt (disk $(df -h /tmp | awk 'NR==2{print $4}') free) $(date '+%H:%M')" | tee -a "$STATE"
  ds=$(echo "$cnt" | grep -oE "DEEPSEEK=[0-9]+" | cut -d= -f2)
  if [ "${ds:-0}" -ge "$TARGET" ]; then
    echo "  TARGET $TARGET reached (DEEPSEEK=$ds)" | tee -a "$STATE"; break
  fi
done
echo "--- gen_more done $(date '+%m-%d %H:%M') $($PY merge_deepseek.py) ---" | tee -a "$STATE"
