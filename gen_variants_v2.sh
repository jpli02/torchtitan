#!/bin/bash
# Fill valid variants across ALL 80 seeds. Workers take seeds round-robin,
# EMPTIES FIRST; for each seed, generate new variants (fresh r2v indices) and
# validate until it has M valid or KMAX attempts are spent. Appends to the
# existing valid/ pool.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
PY=/home/jli199/boptim-agent/.venv/bin/python
SEEDS_DIR=/home/jli199/terminal_bench_eval/tb_tasks
VARDIR=/home/jli199/boptim_scratch/tb_variants
VALID=$VARDIR/valid
GEN=$VARDIR/gen
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PROG=$TMP/gen_variants_v2_progress.txt
M=${M:-3}; KMAX=${KMAX:-8}; CONC=${CONC:-4}; MODEL=${MODEL:-gpt-6-luna}
mkdir -p "$VALID" "$GEN"
: > "$PROG"
echo "[v2] start $(date '+%m-%d %H:%M') M=$M KMAX=$KMAX conc=$CONC" >> "$PROG"

# seeds sorted by current valid-count ascending (empties first)
have_count(){ ls -d "$VALID/$1"__* 2>/dev/null | wc -l; }
mapfile -t SEEDS < <(for d in "$SEEDS_DIR"/*/; do n=$(basename "$d"); echo "$(have_count "$n") $n"; done | sort -n | awk '{print $2}')

worker(){
  local wi=$1 i n have k out res
  for ((i=wi; i<${#SEEDS[@]}; i+=CONC)); do
    n="${SEEDS[$i]}"
    have=$(have_count "$n")
    for ((k=1; k<=KMAX; k++)); do
      [ "$have" -ge "$M" ] && break
      out="$GEN/${n}__r2v${k}"; rm -rf "$out"
      timeout 300 "$PY" "$WT/gen_task_variant.py" --seed "$SEEDS_DIR/$n" --out "$out" --model "$MODEL" --effort high >/dev/null 2>&1 || { echo "$(date '+%H:%M') $n r2v$k GEN_FAIL" >> "$PROG"; continue; }
      res=$(timeout 540 bash "$WT/validate_variant.sh" "$out" 2>/dev/null | head -1)
      echo "$(date '+%H:%M') $n r2v$k -> $res" >> "$PROG"
      if echo "$res" | grep -q "^VALID"; then mv "$out" "$VALID/${n}__r2v${k}"; have=$((have+1)); else rm -rf "$out"; fi
    done
  done
}
for ((w=0; w<CONC; w++)); do worker "$w" & done
wait
echo "[v2] done $(date '+%m-%d %H:%M') valid_total=$(ls -d "$VALID"/*/ 2>/dev/null | wc -l) seeds_covered=$(ls -1 "$VALID" 2>/dev/null | sed -E 's/__r?2?v[0-9]+$//;s/__v[0-9]+$//' | sort -u | wc -l)" >> "$PROG"
