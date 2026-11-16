#!/bin/bash
# Learning-rate debug on GPU 1, part 4. The Terminal-Bench test evaluation of the checkpoint halfway between v3 and r1
# (lerp_r1_a050_eval, started 11-16 03:25 by part 3) is left to finish; then:
#   1. the recipe the held-out curves point to: v3 -> r1 mix at lr 5e-6 with cosine decay to 5e-7, 804 steps (1.5 epochs),
#      validated every 134 steps, checkpoints every 268 steps;
#   2. the Terminal-Bench TEST evaluation of its final checkpoint (11 tasks x 2 attempts, protocol of the epoch 5-8 runs).
# Start detached:  setsid nohup bash lr_debug_chain4.sh >> $TMP/lr_debug_chain4.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
S=/home/jli199/boptim_scratch
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
TAG=lr5e-6_cos804
cd "$WT" || exit 1
END=$(( $(date +%s) + 64800 ))
while pgrep -u "$(id -un)" -f 'eval_gated.sh|ouro_openai_server.py|tb run --dataset-path' > /dev/null \
   || [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | head -n 1)" -lt 44000 ]; do
  [ "$(date +%s)" -gt "$END" ] && { echo "LR-DEBUG part 4 gave up waiting $(date '+%m-%d %H:%M')"; exit 1; }
  sleep 60
done
python3 -I tb_test_summary.py lerp_r1_a050_eval
echo "LR-DEBUG TB evaluation of lerp_r1_a050 done $(date '+%m-%d %H:%M')"

GPU=1 LR=5e-6 STEPS=804 DECAY=0.9 TAG=$TAG CKPT_INTERVAL=268 bash lr_sweep_run.sh
"$PY" -I lr_debug_valcurve.py "$TAG"
echo "LR-DEBUG recipe run done $(date '+%m-%d %H:%M')"
src=$S/ouro_lrsweep_${TAG}/checkpoint/step-804
clean=$S/ouro_lrsweep_${TAG}_clean
if ls "$src"/model*.safetensors > /dev/null 2>&1; then
  case "$clean" in /home/jli199/boptim_scratch/ouro_lrsweep_*_clean) rm -rf "$clean" ;; esac
  mkdir -p "$clean"
  cp "$src"/model*.safetensors "$clean"/
  [ -f "$src/model.safetensors.index.json" ] && cp "$src/model.safetensors.index.json" "$clean"/
  for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$clean/$f"; done
  export OURO_SEQ_CAP=12000
  echo "TB test evaluation of $TAG starts $(date '+%m-%d %H:%M')"
  GPU=1 GATE=0 CKPT="$clean" NAME=${TAG}_eval PORT=8183 ATT=2 MINFREE=${MINFREE:-38000} TASKS="$ALL11" \
    bash eval_gated.sh > "$TMP/${TAG}_eval.log" 2>&1
  python3 -I tb_test_summary.py "${TAG}_eval"
  echo "LR-DEBUG TB evaluation of $TAG done $(date '+%m-%d %H:%M')"
else
  echo "no final safetensors at $src; TB evaluation skipped"
fi
echo "LR-DEBUG part 4 done $(date '+%m-%d %H:%M')"
