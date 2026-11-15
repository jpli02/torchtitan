#!/bin/bash
# GPU 1 lane, fifth version (replaces gpu1_lane3.sh after epoch 8 finished at 46 of 75). Same two protections:
#   * a harness killed from outside leaves tasks unrun  -> leftovers are resumed until all 75 tasks have a result;
#   * a killed harness also leaves its trial container behind, and the harness starts a trial with `docker compose up` under the
#     same project name (task-1-of-1-RUN), so the next trial of that task REUSES the left-over container with the old attempt's
#     files and terminal session in it. Such a trial is not a clean measurement: its result is set aside and the task is run again
#     (the harness removes the container when the dirty trial ends, so the rerun gets a fresh one).
# Order: TEST evaluation of epoch 8 first (the best checkpoint on the training tasks) -> epoch 6 leftovers -> test evaluations of epochs 7 6 5.
# Start detached:  setsid nohup bash gpu1_lane5.sh > $TMP/gpu1_lane5.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
OUT=/srv/scratch/jli199/tb_runs
DUMP=/home/jli199/boptim_scratch/ouro_tvary_r1_8ep
cd "$WT" || exit 1
REP=$TMP/train_tvary_r1_8ep.txt
TASKLIST=$(cat "$TMP/train_tasks_r1.txt")
# containers of run train75_ep8_1800 that were already there when its first pass started (left by the earlier, discarded attempts)
FIRST_DIRTY="swe-bench-fsspec__r2v1 swe-bench-astropy-1__r2v3 grid-pattern-transform__r2v1 fix-pandas-version__r2v1"

count() { python3 -I tb_counts_merged.py "$OUT" "$1" | grep -o 'results=[0-9]*' | cut -d= -f2; }
leftover_containers() {  # RUN ROOT: tasks whose left-over trial container (running or exited) mounts ROOT/RUN/..., so that a new
  local c                # trial under ROOT would reuse it; a container that mounts another root is recreated by compose and is harmless
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
        echo "[lane] $name: $t ran in a left-over container; its result is set aside and the task will be rerun" | tee -a "$REP"
      fi
    done
  done
}
finish() {  # RUN EPOCH PORT: resume until all 75 tasks have a clean result (at most five rounds)
  local name=$1 k=$2 port=$3 try t remaining run_dirty
  for try in 1 2 3 4 5; do
    if [ "$(count "$name")" -ge 75 ]; then echo "[lane] $name complete: $(python3 -I tb_counts_merged.py "$OUT" "$name")" | tee -a "$REP"; return 0; fi
    python3 -I tb_merge_resume.py "$OUT" "$name" | tee -a "$REP"
    remaining=" $(python3 -I tb_counts_merged.py "$OUT" "$name" --remaining $TASKLIST) "
    run_dirty=""
    for t in $(leftover_containers "$name" "$OUT/resume_$name"); do case "$remaining" in *" $t "*) run_dirty="$run_dirty $t" ;; esac; done
    [ -n "$run_dirty" ] && echo "[lane] $name: left-over containers for:$run_dirty (those trials will be redone)" | tee -a "$REP"
    NAME=$name CKPT=${DUMP}_ep${k}_clean GPU=1 PORT=$port bash resume_tb_run2.sh
    [ -n "$run_dirty" ] && set_aside "$name" $run_dirty
  done
  if [ "$(count "$name")" -ge 75 ]; then echo "[lane] $name complete: $(python3 -I tb_counts_merged.py "$OUT" "$name")" | tee -a "$REP"; return 0; fi
  echo "[lane] $name still incomplete after five rounds: $(python3 -I tb_counts_merged.py "$OUT" "$name")" | tee -a "$REP"
}

echo "[lane] epoch 8 test evaluation first $(date '+%m-%d %H:%M')" | tee -a "$REP"
GPU=1 MINFREE=38000 SKIP_SUBSET=1 EPOCHS_LIST="8" bash epoch_eval_chain.sh
echo "[lane] epoch 8 test evaluation returned $? $(date '+%m-%d %H:%M')" | tee -a "$REP"
finish train75_ep6_1800 6 8196
echo "[lane] training-set runs done $(date '+%m-%d %H:%M'); remaining test evaluations start" | tee -a "$REP"
left=$(docker ps -a --format '{{.Names}}' | grep -- '-of-2-tvary_r1_8ep_ep[5-7]_eval$' | tr '\n' ' ')
[ -n "$left" ] && echo "[lane] WARNING: left-over test-evaluation containers, their trials will reuse them and must be redone: $left" | tee -a "$REP"
GPU=1 MINFREE=38000 SKIP_SUBSET=1 EPOCHS_LIST="7 6 5" bash epoch_eval_chain.sh
echo "[lane] test evaluations returned $? $(date '+%m-%d %H:%M')" | tee -a "$REP"
