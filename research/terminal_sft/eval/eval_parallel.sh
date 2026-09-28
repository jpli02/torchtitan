#!/bin/bash
# Task-parallel eval: shard the 12 tasks across every currently-free GPU, one
# ouro server per GPU, then aggregate. A serial $WT/research/terminal_sft/eval/eval_gated.sh run is ~12 tasks x
# 2 attempts on one card; this cuts wall-clock by the number of free GPUs.
#
#   CKPT=<serving dir> BASENAME=tb80sft_v3 GATE=0 MINFREE=28000 bash $WT/research/terminal_sft/eval/eval_parallel.sh
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
: "${CKPT:?serving dir}"; : "${BASENAME:?run name}"
GATE=${GATE:-0}
MINFREE=${MINFREE:-28000}
ATT=${ATT:-2}
cd "$WT" || exit 1
TASKS="hello-world fix-permissions extract-safely csv-to-parquet \
processing-pipeline fix-git heterogeneous-dates chess-best-move \
fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"

# free GPUs (>= MINFREE MiB), as an array
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | tr -d ' ' | awk -F, -v m="$MINFREE" '$2+0>=m {print $1}')
NG=${#GPUS[@]}
OUT=$TMP/${BASENAME}_paralleleval.txt
: > "$OUT"
if [ "$NG" -eq 0 ]; then
  echo "no free GPU (>= ${MINFREE}MiB); falling back to serial on the busiest-wait" | tee -a "$OUT"
  exit 1
fi
echo "[parallel-eval] $NG free GPUs: ${GPUS[*]}  gate=$GATE  $(date '+%m-%d %H:%M')" | tee -a "$OUT"

# round-robin the 12 tasks into NG shards
declare -a SHARD
i=0
for t in $TASKS; do
  k=$((i % NG)); SHARD[$k]="${SHARD[$k]:-} $t"; i=$((i+1))
done

pids=()
for k in $(seq 0 $((NG-1))); do
  g=${GPUS[$k]}; port=$((8120 + k))
  name="${BASENAME}_sh${k}"; [ "$GATE" = "1" ] && name="gated_${BASENAME}_sh${k}"
  echo "  shard $k -> gpu $g port $port tasks:${SHARD[$k]}" | tee -a "$OUT"
  GPU=$g GATE=$GATE CKPT="$CKPT" NAME="$name" PORT=$port ATT=$ATT MINFREE=$MINFREE \
    TASKS="${SHARD[$k]}" bash $WT/research/terminal_sft/eval/eval_gated.sh >> "$TMP/${name}.log" 2>&1 &
  pids+=($!)
done
echo "[parallel-eval] launched ${#pids[@]} shards, waiting..." | tee -a "$OUT"
wait "${pids[@]}"
echo "[parallel-eval] all shards done $(date '+%m-%d %H:%M')" | tee -a "$OUT"

pref="${BASENAME}"; [ "$GATE" = "1" ] && pref="gated_${BASENAME}"
/home/jli199/torchtitan/.venv/bin/python - "$pref" >> "$OUT" 2>&1 <<'PY'
import json, glob, sys, collections
pref = sys.argv[1]
per = collections.defaultdict(lambda: [0, 0])
for f in glob.glob(f"/tmp/tb_runs/{pref}_sh*/*/*/results.json"):
    d = json.load(open(f)); t = d.get("task_id")
    per[t][1] += 1; per[t][0] += 1 if d.get("is_resolved") else 0
tot = sum(v[0] for v in per.values()); n = sum(v[1] for v in per.values())
print(f"{pref}: {tot}/{n} solved   tasks>=1: {sum(1 for v in per.values() if v[0]>0)}/{len(per)}")
for t,(s,c) in sorted(per.items()):
    if s: print(f"   SOLVED {t} {s}/{c}")
print("compare: traj20k 5/24 (54.3%) | v2@10k 4/22 | continue10k 5/24")
PY
cat "$OUT"
