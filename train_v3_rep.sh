#!/bin/bash
# Item 2: v3 replicate from the Ouro-2.6B-Thinking base with the SAME recipe as
# run_26b.sh (20k steps, seq 4096, AdamW8bit, tb80sft v3 data) but a different
# data order (shuffled, seed 1) and RNG seed. Measures the seed/order noise floor
# of the whole pipeline: v3 scored 6/22 stock, 8/22 lever-1, 9/22 gated.
# Then stages the final checkpoint and evals the 11 valid tasks x 2, ungated and
# gated, with the lever-1 serving. Polls for a free GPU other than $SKIPGPU.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
TAG=tb80sft26b_v3rep1
SRCJSONL=/home/jli199/boptim_scratch/tb80sft/train_v3.jsonl
JSONL=$TMP/train_v3_shuf1.jsonl
DUMP=/home/jli199/boptim_scratch/ouro_${TAG}
CLEAN=${DUMP}_clean
REP=$TMP/train_v3_rep.txt
STEPS=${STEPS:-20000}
SKIPGPU=${SKIPGPU:-5}
MINFREE=${MINFREE:-40000}
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
: > "$REP"
echo "=== V3 REPLICATE start $(date '+%m-%d %H:%M') steps=$STEPS ===" | tee -a "$REP"

"$PY" - "$SRCJSONL" "$JSONL" <<'PYEOF'
import sys, random
src, out = sys.argv[1], sys.argv[2]
rows = [l if l.endswith("\n") else l + "\n" for l in open(src) if l.strip()]
random.seed(1); random.shuffle(rows)
open(out, "w").writelines(rows)
print(f"shuffled {len(rows)} rows -> {out}")
PYEOF

gpu=""
for _ in $(seq 1 4320); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' \
        | awk -F, -v m="$MINFREE" -v s="$SKIPGPU" '$1!=s && $2+0>=m {print $2+0, $1}' | sort -rn | head -1 | awk '{print $2}')
  [ -n "$gpu" ] && break
  sleep 60
done
[ -n "$gpu" ] || { echo "ABORT no GPU with ${MINFREE}MiB free (other than $SKIPGPU) in 72h" | tee -a "$REP"; exit 1; }
echo "training on gpu $gpu $(date '+%m-%d %H:%M')" | tee -a "$REP"

cd "$WT" || exit 1
export CUDA_VISIBLE_DEVICES=$gpu
export OURO_SFT_MIX=tb80sft OURO_SFT_LOCAL_JSONL=$JSONL
unset OURO_INIT_FROM OURO_INIT_DCP OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
export WANDB_MODE=offline WANDB_PROJECT=ouro-tb WANDB_RUN_NAME=$TAG WANDB_RUN_GROUP=v3-replicate
export WANDB_DIR=/home/jli199/boptim_scratch/wandb
export PYTORCH_ALLOC_CONF=expandable_segments:True
rm -rf "$DUMP"; mkdir -p "$DUMP"
"$PY" -m torch.distributed.run --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 -m torchtitan.train \
  --module ouro --config ouro_2_6b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps "$STEPS" --training.seq_len 4096 --debug.seed 1 \
  --checkpoint.interval 5000 --metrics.log_freq 50 --metrics.enable_wandb \
  > /home/jli199/terminal_bench_eval/logs/${TAG}_train.log 2>&1
echo "training exited $(date '+%m-%d %H:%M')" | tee -a "$REP"

SRC=$(ls -d "$DUMP"/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$SRC" ] || { echo "ABORT no checkpoint" | tee -a "$REP"; exit 1; }
rm -rf "$CLEAN"; mkdir -p "$CLEAN"
if ls "$SRC"/model*.safetensors >/dev/null 2>&1; then
  cp "$SRC"/model*.safetensors "$CLEAN"/
  [ -f "$SRC/model.safetensors.index.json" ] && cp "$SRC/model.safetensors.index.json" "$CLEAN"/
else
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$WT" "$PY" scripts/export_dcp_to_hf.py \
    --module ouro --config ouro_2_6b_thinking_terminal_sft \
    --checkpoint "$SRC" --hf_template "$BASE" --out "$CLEAN" --dtype bf16 > "$TMP/${TAG}_export.log" 2>&1
  rm -rf "$CLEAN/sharded"
fi
for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$CLEAN/$f"; done
ls "$CLEAN"/model*.safetensors >/dev/null 2>&1 || { echo "ABORT staging produced no weights" | tee -a "$REP"; exit 1; }
echo "staged $CLEAN $(date '+%m-%d %H:%M')" | tee -a "$REP"

export OURO_SEQ_CAP=12000
for gate in 0 1; do
  name=v3rep1_g${gate}
  GPU=$gpu GATE=$gate CKPT="$CLEAN" NAME=$name PORT=8147 ATT=2 MINFREE=40000 TASKS="$ALL11" \
    bash eval_gated.sh > "$TMP/${name}_eval.log" 2>&1
  "$PY" - "$name" >> "$REP" 2>&1 <<'PYEOF'
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
done
echo "compare: v3 (original order) lever-1 8/22 ungated, 9/22 gated" | tee -a "$REP"
echo "=== V3 REPLICATE done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
