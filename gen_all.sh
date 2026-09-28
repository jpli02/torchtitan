#!/bin/bash
# Scaled rejection-sampling generation across the convertible Nemotron
# categories, one at a time with cleanup between so disk stays bounded (the
# 80-task eval shares /tmp, ~2.5G free). For each category: convert N tasks,
# run the strong model, harvest resolved trajectories to a per-category jsonl,
# then prune that category's docker images + tb_runs dir (rows are already
# saved). Appends nothing to disk that harvest_traj.py hasn't captured.
#
# Categories: the ones Nemotron actually has AND that fill/help TB-80.
# security is the biggest gap; sysadmin-proper, model-training and games are
# NOT in Nemotron and are intentionally absent here.
#
# Env: CATS  N(=40)  ATT(=2)  CONC(=3)  MODEL(=openai/gpt-5-mini)
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
OUTDIR=/home/jli199/boptim_scratch/gen_out
GENTASKS=/home/jli199/boptim_scratch/gen_tasks
CATS=${CATS:-"security debugging data_science file_operations scientific_computing"}
N=${N:-40}; ATT=${ATT:-2}; CONC=${CONC:-3}; MODEL=${MODEL:-openai/gpt-5-mini}
STATE=$TMP/gen_all_state.txt; : > "$STATE"
mkdir -p "$OUTDIR"
cd "$WT" || exit 1

echo "[gen_all] cats=[$CATS] N=$N att=$ATT model=$MODEL $(date '+%m-%d %H:%M')" | tee -a "$STATE"
for cat in $CATS; do
  echo "=== $cat: convert $N ($(date '+%H:%M')) ===" | tee -a "$STATE"
  rm -rf "$GENTASKS/$cat"
  /home/jli199/torchtitan/.venv/bin/python convert_harbor.py --category "$cat" --n "$N" \
    --out "$GENTASKS" >> "$STATE" 2>&1
  rid="genall_${cat}"
  TASKS_DIR="$GENTASKS/$cat" RUNID="$rid" MODEL="$MODEL" ATT="$ATT" CONC="$CONC" \
    bash gen_traj.sh >> "$STATE" 2>&1
  /home/jli199/torchtitan/.venv/bin/python harvest_traj.py "$rid" \
    --out "$OUTDIR/$cat.jsonl" >> "$STATE" 2>&1
  # disk hygiene: drop this run's trajectories + casts + built images
  find /tmp/tb_runs/$rid -name "*.cast" -delete 2>/dev/null
  rm -rf /tmp/tb_runs/$rid 2>/dev/null
  docker image prune -f >/dev/null 2>&1
  echo "  $cat done; disk $(df -h /tmp | awk 'NR==2{print $4}') free" | tee -a "$STATE"
done

echo "=== combine ===" | tee -a "$STATE"
cat "$OUTDIR"/*.jsonl > "$OUTDIR/generated_all.jsonl" 2>/dev/null
n=$(wc -l < "$OUTDIR/generated_all.jsonl" 2>/dev/null || echo 0)
echo "TOTAL harvested rows: $n -> $OUTDIR/generated_all.jsonl" | tee -a "$STATE"
echo "--- gen_all done $(date '+%m-%d %H:%M') ---" | tee -a "$STATE"
