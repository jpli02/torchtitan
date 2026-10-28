#!/bin/bash
# Item 1: take-turns r1 at a TRUE 4 epochs (steps sized by tokens), shuffled mix,
# checkpoint every epoch, then export + 11-task eval (lever-1 serving, ungated,
# pass@2) for each epoch checkpoint. Compare: r1 at 1.67 epochs = 9/22 (6 tasks),
# v3+lever-1 8/22, baseline 6/22.
#
# Token budget (render_sft_rows.py): tvary_r1_mixed = 224 rows, 2,194,525 tokens
# -> 1 epoch = 536 steps of 4096 packed tokens; 4 epochs = 2144 steps.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
INIT=/home/jli199/boptim_scratch/ouro_tb80sft26b_v3/checkpoint/step-20000
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
TAG=ouro_tvary_r1_4ep
DUMP=/home/jli199/boptim_scratch/${TAG}
MIX=$TMP/tvary_r1_mixed.jsonl
SHUF=$TMP/tvary_r1_mixed_shuf1.jsonl
REP=$TMP/train_tvary_r1_4ep.txt
GPU=${GPU:-5}
SPE=536; EPOCHS=4; STEPS=$((SPE*EPOCHS))
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
: > "$REP"
echo "=== TRAIN_TVARY r1 4ep start $(date '+%m-%d %H:%M') gpu=$GPU steps=$STEPS (spe=$SPE) ===" | tee -a "$REP"

free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | awk -F, -v g="$GPU" '$1==g {print $2}')
[ "${free:-0}" -ge 40000 ] || { echo "ABORT gpu $GPU has ${free:-?}MiB free" | tee -a "$REP"; exit 1; }

"$PY" - "$MIX" "$SHUF" <<'PYEOF'
import sys, random
src, out = sys.argv[1], sys.argv[2]
rows = [l if l.endswith("\n") else l + "\n" for l in open(src) if l.strip()]
random.seed(1); random.shuffle(rows)
open(out, "w").writelines(rows)
print(f"shuffled {len(rows)} rows -> {out}")
PYEOF

cd "$WT" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU
export OURO_SFT_MIX=tb80sft OURO_SFT_LOCAL_JSONL=$SHUF
unset OURO_INIT_FROM OURO_INIT_DCP OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
export WANDB_MODE=offline WANDB_PROJECT=ouro-tb WANDB_RUN_NAME=$TAG WANDB_RUN_GROUP=take-turns-4ep
export WANDB_DIR=/home/jli199/boptim_scratch/wandb
export PYTORCH_ALLOC_CONF=expandable_segments:True
rm -rf "$DUMP"; mkdir -p "$DUMP"
echo "training on gpu $GPU $(date '+%m-%d %H:%M')" | tee -a "$REP"
"$PY" -m torch.distributed.run --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 -m torchtitan.train \
  --module ouro --config ouro_2_6b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --checkpoint.initial_load_path "$INIT" --checkpoint.initial_load_in_hf \
  --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps "$STEPS" --training.seq_len 4096 \
  --checkpoint.interval "$SPE" --metrics.log_freq 10 --metrics.enable_wandb \
  > /home/jli199/terminal_bench_eval/logs/${TAG}_train.log 2>&1
echo "training exited $(date '+%m-%d %H:%M'); checkpoints: $(ls -d "$DUMP"/checkpoint/step-* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')" | tee -a "$REP"

stage() {  # stage <step-dir> <clean-dir>: interval saves are DCP -> export; final save is safetensors
  local src=$1 clean=$2
  rm -rf "$clean"; mkdir -p "$clean"
  if ls "$src"/model*.safetensors >/dev/null 2>&1; then
    cp "$src"/model*.safetensors "$clean"/
    [ -f "$src/model.safetensors.index.json" ] && cp "$src/model.safetensors.index.json" "$clean"/
  else
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$WT" "$PY" scripts/export_dcp_to_hf.py \
      --module ouro --config ouro_2_6b_thinking_terminal_sft \
      --checkpoint "$src" --hf_template "$BASE" --out "$clean" --dtype bf16 \
      > "$TMP/${TAG}_export_$(basename "$src").log" 2>&1
    rm -rf "$clean/sharded"
  fi
  for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$clean/$f"; done
  ls "$clean"/model*.safetensors >/dev/null 2>&1
}

export OURO_SEQ_CAP=12000
for k in $(seq 1 "$EPOCHS"); do
  step=$((SPE*k)); src="$DUMP/checkpoint/step-$step"; clean="${DUMP}_ep${k}_clean"
  [ -d "$src" ] || { echo "epoch $k: no checkpoint at $src, skipping" | tee -a "$REP"; continue; }
  if ! stage "$src" "$clean"; then echo "epoch $k: staging FAILED" | tee -a "$REP"; continue; fi
  echo "epoch $k (step $step) staged; eval starting $(date '+%m-%d %H:%M')" | tee -a "$REP"
  GPU=$GPU GATE=0 CKPT="$clean" NAME=tvary_r1_4ep_ep${k} PORT=8146 ATT=2 MINFREE=40000 \
    TASKS="$ALL11" bash eval_gated.sh > "$TMP/tvary_r1_4ep_ep${k}_eval.log" 2>&1
  "$PY" - tvary_r1_4ep_ep${k} >> "$REP" 2>&1 <<'PYEOF'
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
  echo "epoch $k eval done $(date '+%m-%d %H:%M')" | tee -a "$REP"
done
echo "=== summary (11 valid tasks x 2, lever-1 ungated) ===" | tee -a "$REP"
for k in $(seq 1 "$EPOCHS"); do grep -E "^tvary_r1_4ep_ep${k}: " "$REP" | tail -1; done | tee -a "$REP"
echo "compare: r1 @1.67ep 9/22 (6 tasks) | v3+lever-1 8/22 | baseline 6/22" | tee -a "$REP"
echo "=== TRAIN_TVARY r1 4ep done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
