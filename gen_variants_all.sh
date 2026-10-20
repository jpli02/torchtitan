#!/bin/bash
# Generate + validate TB task variants across the 80 seed tasks. Each worker
# takes seeds round-robin, generates up to K variants per seed (gpt-6-luna),
# validates each (fresh=FAIL, solved=PASS), and moves VALID ones into the pool.
# Stops when the valid pool reaches TARGET or seeds are exhausted.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
PY=/home/jli199/boptim-agent/.venv/bin/python
SEEDS_DIR=/home/jli199/terminal_bench_eval/tb_tasks
VARDIR=/home/jli199/boptim_scratch/tb_variants
VALID=$VARDIR/valid
GEN=$VARDIR/gen
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PROG=$TMP/gen_variants_progress.txt
K=${K:-3}
TARGET=${TARGET:-200}
CONC=${CONC:-4}
MODEL=${MODEL:-gpt-6-luna}
mkdir -p "$VALID" "$GEN"
: > "$PROG"
echo "[gen_variants] start $(date '+%m-%d %H:%M') K=$K target=$TARGET conc=$CONC model=$MODEL" >> "$PROG"

mapfile -t SEEDS < <(ls -d "$SEEDS_DIR"/*/ 2>/dev/null)
nvalid(){ ls -d "$VALID"/*/ 2>/dev/null | wc -l; }

worker(){  # $1 = worker index (0..CONC-1)
  local wi=$1 i seed name k out res
  for ((i=wi; i<${#SEEDS[@]}; i+=CONC)); do
    [ "$(nvalid)" -ge "$TARGET" ] && return
    seed="${SEEDS[$i]}"; name=$(basename "$seed")
    for k in $(seq 1 "$K"); do
      [ "$(nvalid)" -ge "$TARGET" ] && return
      out="$GEN/${name}__v${k}"; rm -rf "$out"
      timeout 300 "$PY" "$WT/gen_task_variant.py" --seed "$seed" --out "$out" --model "$MODEL" --effort high >/dev/null 2>&1 || { echo "$(date '+%H:%M') $name v$k GEN_FAIL" >> "$PROG"; continue; }
      res=$(timeout 540 bash "$WT/validate_variant.sh" "$out" 2>/dev/null | head -1)
      echo "$(date '+%H:%M') $name v$k -> $res" >> "$PROG"
      if echo "$res" | grep -q "^VALID"; then mv "$out" "$VALID/${name}__v${k}"; else rm -rf "$out"; fi
    done
  done
}

for ((w=0; w<CONC; w++)); do worker "$w" & done
wait
echo "[gen_variants] done $(date '+%m-%d %H:%M') valid=$(nvalid)" >> "$PROG"
