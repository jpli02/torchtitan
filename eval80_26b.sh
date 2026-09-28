#!/bin/bash
# Full 80-task eval of ONE model, one ouro server per GPU across a pinned GPU
# set, 80 tasks round-robin sharded, 1 attempt each (the n/80 protocol). Used
# for the 2.6B baseline vs SFT comparison; the 2.6B server is bigger than the
# 1.4B, so one server per 45GB card (no multi-server-per-GPU OOM risk).
#
#   MODEL_NAME=base26b CKPT=/path GPUS="0,1,2,3,4" bash eval80_26b.sh
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
TBTASKS=/home/jli199/terminal_bench_eval/tb_tasks
NAME=${MODEL_NAME:?set MODEL_NAME}
CKPT=${CKPT:?set CKPT}
GPUS=${GPUS:?comma-separated GPU set}
BASEPORT=${BASEPORT:-8200}
OUT=$TMP/eval80_${NAME}.txt
: > "$OUT"
cd "$WT" || exit 1

mapfile -t TASKS < <(ls -d "$TBTASKS"/*/ | xargs -n1 basename | sort)
IFS=',' read -ra GARR <<< "$GPUS"
NG=${#GARR[@]}
echo "[$NAME] ${#TASKS[@]} tasks across $NG GPUs ($GPUS) $(date '+%m-%d %H:%M')" | tee -a "$OUT"

# round-robin tasks into NG shards
declare -a SHARD
for i in "${!TASKS[@]}"; do
  k=$(( i % NG )); SHARD[$k]="${SHARD[$k]:-} ${TASKS[$i]}"
done

pids=()
for k in $(seq 0 $((NG-1))); do
  g=${GARR[$k]}; port=$((BASEPORT + k))
  GPU=$g GATE=0 CKPT="$CKPT" NAME="${NAME}_sh${k}" PORT=$port ATT=1 MINFREE=20000 \
    TASKS="${SHARD[$k]}" bash eval_gated.sh >> "$TMP/${NAME}_sh${k}.log" 2>&1 &
  pids+=($!)
done
echo "[$NAME] launched $NG shards, waiting..." | tee -a "$OUT"
wait "${pids[@]}"
echo "[$NAME] all shards done $(date '+%m-%d %H:%M')" | tee -a "$OUT"

/home/jli199/torchtitan/.venv/bin/python - "$NAME" >> "$OUT" 2>&1 <<'PY'
import json, glob, sys, collections
name = sys.argv[1]
per = collections.defaultdict(lambda: [0, 0]); tp = tn = 0
for f in glob.glob(f"/tmp/tb_runs/{name}_sh*/*/*/results.json"):
    d = json.load(open(f)); t = d.get("task_id")
    per[t][1] += 1; per[t][0] += 1 if d.get("is_resolved") else 0
    pr = d.get("parser_results") or {}
    if isinstance(pr, dict) and pr:
        tn += len(pr); tp += sum(1 for v in pr.values() if str(v).lower() in ("passed","true","ok"))
tot = sum(v[0] for v in per.values()); n = sum(v[1] for v in per.values())
print(f"{name}: {tot}/{n} solved   tasks>=1: {sum(1 for v in per.values() if v[0]>0)}/{len(per)}")
for t,(s,c) in sorted(per.items()):
    if s: print(f"   SOLVED {t} {s}/{c}")
print(f"   partial credit: {tp}/{tn} ({100*tp/max(tn,1):.1f}%)")
PY
cat "$OUT"
