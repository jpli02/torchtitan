#!/bin/bash
# GPU 1 lane, add-on behind gpu1_lane3.sh: the checkpoints that were never scored on the full 75 training tasks, so that the
# per-checkpoint table has every row (same protocol as the others: one attempt, 1,800 s, 2,048-token turn cap, 12k context cap):
#   v3 (the starting checkpoint), then epochs 2, 1, 3 of the first four-epoch run; then the epoch 3 test evaluation.
# Each run goes through the leftover-resume loop, so a killed harness or a left-over container costs a rerun, not the result.
# Start detached:  setsid nohup bash gpu1_lane4.sh > $TMP/gpu1_lane4.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
OUT=/srv/scratch/jli199/tb_runs
PY=/home/jli199/torchtitan/.venv/bin/python
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
RUN4=/home/jli199/boptim_scratch/ouro_tvary_r1_4ep
cd "$WT" || exit 1
REP=$TMP/train_tvary_r1_8ep.txt
TASKLIST=$(cat "$TMP/train_tasks_r1.txt")

count() { python3 -I tb_counts_merged.py "$OUT" "$1" | grep -o 'results=[0-9]*' | cut -d= -f2; }
leftover_containers() {  # RUN ROOT: tasks whose left-over trial container mounts ROOT/RUN/... (a new trial under ROOT would reuse it)
  local c
  for c in $(docker ps -a --format '{{.Names}}' | grep -- "-1-of-1-$1\$"); do
    case "$(docker inspect -f '{{range .Mounts}}{{.Source}} {{end}}' "$c" 2>/dev/null)" in
      "$2/$1/"*) echo "${c%-1-of-1-$1}" ;;
    esac
  done
}
set_aside() {  # RUN task...: move finished trials that ran in a left-over container out of the run's roots
  local name=$1 t root; shift
  mkdir -p "$OUT/${name}_dirty_container"
  for t in "$@"; do
    for root in "$OUT/$name" "$OUT/resume_$name/$name"; do
      if ls "$root/$t"/*/results.json > /dev/null 2>&1; then
        mv "$root/$t" "$OUT/${name}_dirty_container/$t.$(date +%s)"
        echo "[lane4] $name: $t ran in a left-over container; its result is set aside and the task will be rerun" | tee -a "$REP"
      fi
    done
  done
}
run75() {  # RUN CKPT PORT: all 75 training tasks, resumed until every task has a clean result (at most five rounds)
  local name=$1 ckpt=$2 port=$3 try t remaining run_dirty
  ls "$ckpt"/model*.safetensors > /dev/null 2>&1 || { echo "[lane4] $name: no weights in $ckpt, skipped" | tee -a "$REP"; return 1; }
  echo "[lane4] $name starts $(date '+%m-%d %H:%M') ($ckpt)" | tee -a "$REP"
  for try in 1 2 3 4 5; do
    if [ "$(count "$name")" -ge 75 ]; then break; fi
    python3 -I tb_merge_resume.py "$OUT" "$name" | tee -a "$REP"
    remaining=" $(python3 -I tb_counts_merged.py "$OUT" "$name" --remaining $TASKLIST) "
    run_dirty=""
    for t in $(leftover_containers "$name" "$OUT/resume_$name"); do case "$remaining" in *" $t "*) run_dirty="$run_dirty $t" ;; esac; done
    [ -n "$run_dirty" ] && echo "[lane4] $name: left-over containers for:$run_dirty (those trials will be redone)" | tee -a "$REP"
    NAME=$name CKPT=$ckpt GPU=1 PORT=$port bash resume_tb_run2.sh
    [ -n "$run_dirty" ] && set_aside "$name" $run_dirty
  done
  python3 -I tb_merge_resume.py "$OUT" "$name" | tee -a "$REP"
  echo "[lane4] $name ended $(date '+%m-%d %H:%M'): $(python3 -I tb_counts_merged.py "$OUT" "$name")" | tee -a "$REP"
}
stage3() {  # epoch 3 of the four-epoch run was never exported: interval save (DCP) -> serving directory
  local src=$RUN4/checkpoint/step-1608 clean=${RUN4}_ep3_clean f
  ls "$clean"/model*.safetensors > /dev/null 2>&1 && return 0
  rm -rf "$clean"; mkdir -p "$clean"
  if ls "$src"/model*.safetensors > /dev/null 2>&1; then
    cp "$src"/model*.safetensors "$clean"/
    [ -f "$src/model.safetensors.index.json" ] && cp "$src/model.safetensors.index.json" "$clean"/
  else
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$WT" "$PY" scripts/export_dcp_to_hf.py --module ouro --config ouro_2_6b_thinking_terminal_sft \
      --checkpoint "$src" --hf_template "$BASE" --out "$clean" --dtype bf16 > "$TMP/ouro_tvary_r1_4ep_export_step-1608.log" 2>&1 || return 1
    [ -d "$clean/sharded" ] && rm -rf "$clean/sharded"
  fi
  for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$clean/$f"; done
  ls "$clean"/model*.safetensors > /dev/null 2>&1
}

until grep -q '\[lane\] test evaluations returned' "$REP" 2>/dev/null; do sleep 60; done
echo "[lane4] add-on starts $(date '+%m-%d %H:%M')" | tee -a "$REP"
run75 train75_v3_1800 /home/jli199/boptim_scratch/ouro_tb80sft26b_v3_clean 8191
run75 train75_ep2_1800 ${RUN4}_ep2_clean 8192
run75 train75_ep1_1800 ${RUN4}_ep1_clean 8193
if stage3; then
  run75 train75_ep3_1800 ${RUN4}_ep3_clean 8194
  # epoch 3 is also the one checkpoint of the first run without a test score: 11 tasks x 2 attempts, the settings of the other epochs
  ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
  OURO_SEQ_CAP=12000 GPU=1 GATE=0 CKPT=${RUN4}_ep3_clean NAME=tvary_r1_4ep_ep3_eval PORT=8163 ATT=2 MINFREE=38000 TASKS="$ALL11" \
    bash eval_gated.sh > "$TMP/tvary_r1_4ep_ep3_eval.log" 2>&1
  echo "[lane4] epoch 3 test evaluation ended $(date '+%m-%d %H:%M'): $(find /tmp/tb_runs/tvary_r1_4ep_ep3_eval -name results.json -path '*/*/*' 2>/dev/null | wc -l) trials" | tee -a "$REP"
else
  echo "[lane4] epoch 3: staging failed" | tee -a "$REP"
fi
echo "[lane4] add-on finished $(date '+%m-%d %H:%M')" | tee -a "$REP"
