#!/bin/bash
# For each epoch k in EPOCHS_LIST (5..8 of the ep4 continuation): wait for its checkpoint, stage it to an HF serving dir, then run
#   (a) the 11-task Terminal-Bench test eval, 2 attempts, lever-1 serving (same protocol as the ep1..ep4 clean evals; 5/21 for ep4)
#   (b) the fixed 25-task training subset (train_tasks_sub25.txt: every third of the 75; ep4 8/25, r1 7/25), 1 attempt, 1,800 s
# on one GPU, keeping the card busy from one step to the next.
#   GPU=3 EPOCHS_LIST="5 7" bash epoch_eval_chain.sh
set -u
: "${GPU:?}"; : "${EPOCHS_LIST:?}"
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
TAG=ouro_tvary_r1_8ep; DUMP=/home/jli199/boptim_scratch/$TAG; SPE=536
POOL=/home/jli199/boptim_scratch/tb_variants/pool; OUT=/srv/scratch/jli199/tb_runs
REP=$TMP/train_tvary_r1_8ep.txt
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
SUB25=$(cat "$TMP/train_tasks_sub25.txt")
export OURO_SEQ_CAP=12000   # the context cap of every epoch 1..4 evaluation; without it contexts grow past 16k tokens and the prefill runs out of memory
cd "$WT" || exit 1
stage() {  # stage <step-dir> <clean-dir>: interval saves are DCP -> export; a final save is safetensors
  local src=$1 clean=$2
  rm -rf "$clean"; mkdir -p "$clean"
  if ls "$src"/model*.safetensors >/dev/null 2>&1; then
    cp "$src"/model*.safetensors "$clean"/
    [ -f "$src/model.safetensors.index.json" ] && cp "$src/model.safetensors.index.json" "$clean"/
  else
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$WT" "$PY" scripts/export_dcp_to_hf.py \
      --module ouro --config ouro_2_6b_thinking_terminal_sft \
      --checkpoint "$src" --hf_template "$BASE" --out "$clean" --dtype bf16 > "$TMP/${TAG}_export_$(basename "$src").log" 2>&1
    rm -rf "$clean/sharded"
  fi
  for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$clean/$f"; done
  ls "$clean"/model*.safetensors >/dev/null 2>&1
}
for k in $EPOCHS_LIST; do
  step=$(( SPE * (k - 4) )); src="$DUMP/checkpoint/step-$step"; clean="${DUMP}_ep${k}_clean"
  echo "[gpu $GPU] waiting for epoch $k (step $step) $(date '+%m-%d %H:%M')"
  until [ -d "$src" ]; do [ -f "$DUMP/TRAIN_DONE" ] && [ ! -d "$src" ] && { echo "epoch $k: training finished without $src" | tee -a "$REP"; break; }; sleep 60; done
  [ -d "$src" ] || continue
  # the save is complete when nothing in it changed for 2 minutes
  while [ -n "$(find "$src" -newermt '-120 seconds' 2>/dev/null | head -n 1)" ]; do sleep 30; done
  if ! stage "$src" "$clean"; then echo "epoch $k: staging FAILED" | tee -a "$REP"; continue; fi
  echo "epoch $k (step $step) staged $clean; evals on gpu $GPU start $(date '+%m-%d %H:%M')" | tee -a "$REP"
  # (a) test set (skipped when 22 results already exist from a clean run)
  if [ "$(find /tmp/tb_runs/tvary_r1_8ep_ep${k}_eval -name results.json 2>/dev/null | wc -l)" -ge 22 ]; then
    echo "epoch $k test eval already complete; skipping" | tee -a "$REP"
  else
  GPU=$GPU GATE=0 CKPT="$clean" NAME=tvary_r1_8ep_ep${k}_eval PORT=$((8160 + k)) ATT=2 MINFREE=${MINFREE:-38000} TASKS="$ALL11" \
    bash eval_gated.sh > "$TMP/tvary_r1_8ep_ep${k}_eval.log" 2>&1
  echo "epoch $k test eval: $(grep -c 'OOM at step' /home/jli199/terminal_bench_eval/logs/tvary_r1_8ep_ep${k}_eval_server.log 2>/dev/null) out-of-memory turns of $(grep -c 'TURN prompt' /home/jli199/terminal_bench_eval/logs/tvary_r1_8ep_ep${k}_eval_server.log 2>/dev/null)" | tee -a "$REP"
  fi
  "$PY" - tvary_r1_8ep_ep${k}_eval >> "$REP" 2>&1 <<'PYEOF'
import json, glob, sys, collections
name=sys.argv[1]; per=collections.defaultdict(lambda:[0,0]); fm=collections.Counter()
for f in glob.glob(f"/tmp/tb_runs/{name}/*/*/results.json"):
    d=json.load(open(f)); t=d.get("task_id"); per[t][1]+=1; per[t][0]+= 1 if d.get("is_resolved") else 0
    fm[str(d.get("failure_mode") or ("RESOLVED" if d.get("is_resolved") else "other"))]+=1
tot=sum(v[0] for v in per.values()); n=sum(v[1] for v in per.values())
print(f"{name}: {tot}/{n} solved  tasks>=1: {sum(1 for v in per.values() if v[0]>0)}/{len(per)}")
for t,(s,c) in sorted(per.items()): print(f"   {'SOLVED' if s else '  --  '} {t} {s}/{c}")
print("failure_modes:", dict(fm))
PYEOF
  # (b) training subset (skipped when 25 results already exist from a clean run)
  if [ "$(find $OUT/train25_ep${k}_1800 -name results.json 2>/dev/null | wc -l)" -ge 25 ]; then
    echo "epoch $k training-subset eval already complete; skipping" | tee -a "$REP"
  else
  OURO_SEQ_CAP=12000 GPU=$GPU GATE=0 MAXTOK=2048 TBTIMEOUT=64800 CKPT="$clean" NAME=train25_ep${k}_1800 PORT=$((8170 + k)) ATT=1 MINFREE=${MINFREE:-38000} \
    DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=1800 TASKS="$SUB25" bash eval_gated.sh > "$TMP/train25_ep${k}_1800.log" 2>&1
  echo "epoch $k training-subset eval: $(grep -c 'OOM at step' /home/jli199/terminal_bench_eval/logs/train25_ep${k}_1800_server.log 2>/dev/null) out-of-memory turns of $(grep -c 'TURN prompt' /home/jli199/terminal_bench_eval/logs/train25_ep${k}_1800_server.log 2>/dev/null)" | tee -a "$REP"
  fi
  python3 -I tb_progress.py "$OUT" train25_ep${k}_1800 | tee -a "$REP"
  echo "epoch $k evals done $(date '+%m-%d %H:%M')" | tee -a "$REP"
done
echo "[gpu $GPU] chain for epochs [$EPOCHS_LIST] finished $(date '+%m-%d %H:%M')" | tee -a "$REP"
