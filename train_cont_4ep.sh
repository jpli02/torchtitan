#!/bin/bash
# Generic v3 continuation at a TRUE number of epochs (steps sized by tokens).
#   DATA=<new rows jsonl> TAG=<run tag> GPU=<idx> [EPOCHS=4] [GATE=0] [PORT=8148] bash train_cont_4ep.sh
# Mixes DATA 1:1 with a random v3 replay sample, shuffles, renders to count tokens,
# sets steps = ceil(EPOCHS * tokens / 4096), continues v3 weights-only (AdamW8bit,
# seq 4096, wandb offline), stages the final model-only save, evals the 11 valid
# tasks x 2 with lever-1 serving, and appends the per-task result to the report.
set -u
: "${DATA:?}"; : "${TAG:?}"; : "${GPU:?}"
EPOCHS=${EPOCHS:-4}; GATE=${GATE:-0}; PORT=${PORT:-8148}; REPLAY=${REPLAY:-1}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
INIT=/home/jli199/boptim_scratch/ouro_tb80sft26b_v3/checkpoint/step-20000
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
V3DATA=/home/jli199/boptim_scratch/tb80sft/train_v3.jsonl
DUMP=/home/jli199/boptim_scratch/${TAG}; CLEAN=${DUMP}_clean
MIX=$TMP/${TAG}_mixed.jsonl; REP=$TMP/train_${TAG}.txt
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
: > "$REP"
cd "$WT" || exit 1
echo "=== TRAIN $TAG start $(date '+%m-%d %H:%M') data=$DATA rows=$(wc -l < "$DATA") epochs=$EPOCHS gpu=$GPU ===" | tee -a "$REP"

"$PY" - "$DATA" "$V3DATA" "$MIX" "$REPLAY" <<'PYEOF'
import sys, random
new, v3, out, replay = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
nl = [l if l.endswith("\n") else l + "\n" for l in open(new) if l.strip()]
vl = [l if l.endswith("\n") else l + "\n" for l in open(v3) if l.strip()]
random.seed(1)
rep = random.sample(vl, min(len(nl), len(vl))) if replay else []
rows = nl + rep
random.shuffle(rows)
open(out, "w").writelines(rows)
print(f"mixed: new={len(nl)} replay={len(rep)} total={len(rows)} (shuffled)")
PYEOF

TOK=$("$PY" -u render_sft_rows.py "$V3DATA" "$MIX" "$TMP/${TAG}_rendered.jsonl" 0 2>/dev/null | grep -oE 'total_tokens=[0-9,]+' | tr -d ',' | cut -d= -f2)
[ -n "${TOK:-}" ] && [ "$TOK" -gt 0 ] || { echo "ABORT token count failed" | tee -a "$REP"; exit 1; }
STEPS=${STEPS:-$(( (EPOCHS*TOK + 4095) / 4096 ))}
echo "tokens=$TOK steps=$STEPS (= $(( STEPS*4096/TOK )) true epochs; replay=$REPLAY)" | tee -a "$REP"

free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | awk -F, -v g="$GPU" '$1==g {print $2}')
[ "${free:-0}" -ge 40000 ] || { echo "ABORT gpu $GPU has ${free:-?}MiB free" | tee -a "$REP"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU
export OURO_SFT_MIX=tb80sft OURO_SFT_LOCAL_JSONL=$MIX
unset OURO_INIT_FROM OURO_INIT_DCP OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
export WANDB_MODE=offline WANDB_PROJECT=ouro-tb WANDB_RUN_NAME=$TAG WANDB_RUN_GROUP=composition
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
  --checkpoint.interval 100000 --metrics.log_freq 10 --metrics.enable_wandb \
  > /home/jli199/terminal_bench_eval/logs/${TAG}_train.log 2>&1
echo "training exited $(date '+%m-%d %H:%M')" | tee -a "$REP"

SRC=$(ls -d "$DUMP"/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$SRC" ] || { echo "ABORT no checkpoint" | tee -a "$REP"; exit 1; }
rm -rf "$CLEAN"; mkdir -p "$CLEAN"
if ls "$SRC"/model*.safetensors >/dev/null 2>&1; then
  cp "$SRC"/model*.safetensors "$CLEAN"/
  [ -f "$SRC/model.safetensors.index.json" ] && cp "$SRC/model.safetensors.index.json" "$CLEAN"/
else
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$WT" "$PY" scripts/export_dcp_to_hf.py \
    --module ouro --config ouro_2_6b_thinking_terminal_sft \
    --checkpoint "$SRC" --hf_template "$BASE" --out "$CLEAN" --dtype bf16 > "$TMP/${TAG}_export.log" 2>&1
  rm -rf "$CLEAN/sharded"
fi
for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$CLEAN/$f"; done
ls "$CLEAN"/model*.safetensors >/dev/null 2>&1 || { echo "ABORT staging produced no weights" | tee -a "$REP"; exit 1; }
echo "staged $CLEAN" | tee -a "$REP"

export OURO_SEQ_CAP=12000
NAME=${TAG#ouro_}_eval
GPU=$GPU GATE=$GATE CKPT="$CLEAN" NAME=$NAME PORT=$PORT ATT=2 MINFREE=40000 TASKS="$ALL11" \
  bash eval_gated.sh > "$TMP/${NAME}.log" 2>&1
"$PY" - "$NAME" >> "$REP" 2>&1 <<'PYEOF'
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
echo "=== TRAIN $TAG done $(date '+%m-%d %H:%M') ===" | tee -a "$REP"
