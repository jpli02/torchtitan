#!/bin/bash
# Multi-teacher CIFAR trajectory generation. Round-robins 3 teachers across a
# pool of GPUs, each teacher to TARGET trajectories, all at high reasoning
# effort, MAXIT iters. One fresh process per trajectory (no CUDA context reuse).
# deepseek routes through OpenRouter; gpt-6-luna / gpt-6.1-sol via OpenAI.
set -u
REPO=/home/jli199/boptim-agent
PY=$REPO/.venv/bin/python
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
WORK=/home/jli199/boptim_scratch/teacher_work
BASE=/home/jli199/boptim_scratch
PROG=$TMP/gen_teachers_progress.txt
GPUS="${GPUS:-8 9}"
TARGET=${TARGET:-100}
MAXIT=${MAXIT:-10}

# load keys WITHOUT echoing (values stay in vars, never printed / never in argv)
set -a; . ~/.boptim_keys.env 2>/dev/null; set +a
OAI_KEY="${OPENAI_API_KEY:-}"
OR_KEY="${OPENROUTER_API_KEY:-}"

declare -A MODEL OUT ROUTE SEED
MODEL[luna]="gpt-6-luna";                  OUT[luna]="$BASE/cifar_gpt6_luna";    ROUTE[luna]=openai;     SEED[luna]=31000
MODEL[sol]="gpt-6.1-sol";                  OUT[sol]="$BASE/cifar_gpt61_sol";     ROUTE[sol]=openai;      SEED[sol]=41000
MODEL[ds]="deepseek/deepseek-v4.1-flash";  OUT[ds]="$BASE/cifar_deepseek_v41";   ROUTE[ds]=openrouter;   SEED[ds]=51000
TEACHERS="luna sol ds"
for t in $TEACHERS; do mkdir -p "${OUT[$t]}"; done

count(){ ls "${OUT[$1]}"/*.json 2>/dev/null | wc -l; }
inflight(){ local t=$1 n=0 g; for g in $GPUS; do [ "${SLOTT[$g]:-}" = "$t" ] && kill -0 "${PID[$g]:-0}" 2>/dev/null && n=$((n+1)); done; echo $n; }

declare -A PID SLOTT
ROT=(luna sol ds); RI=0; PICKED=""
pick(){ # sets global PICKED to next rotation teacher whose completed+inflight < TARGET (RI must persist, so NOT run in $())
  local tries=0 t
  PICKED=""
  while [ $tries -lt 3 ]; do
    t=${ROT[$((RI % 3))]}; RI=$((RI+1)); tries=$((tries+1))
    if [ "$(( $(count "$t") + $(inflight "$t") ))" -lt "$TARGET" ]; then PICKED="$t"; return; fi
  done
}
LAST_PID=0
launch(){ # $1=gpu $2=teacher ; sets LAST_PID, bg (no command subst)
  local g=$1 t=$2 s=${SEED[$2]} wd="$WORK/g$1"
  SEED[$2]=$((s+1)); mkdir -p "$wd"; cd "$wd"
  export CUDA_VISIBLE_DEVICES=$g BOPTIM_REPO=$REPO GPT_REASONING_EFFORT=high
  if [ "${ROUTE[$t]}" = openrouter ]; then
    export OPENAI_API_KEY="$OR_KEY" OPENAI_BASE_URL="https://openrouter.ai/api/v1"
  else
    export OPENAI_API_KEY="$OAI_KEY"; unset OPENAI_BASE_URL
  fi
  nohup "$PY" "$REPO/gen_bo_traj.py" --objective cifar --direction maximize --optim chatgpt \
    --model "${MODEL[$t]}" --n 1 --seed_start "$s" --max_iter "$MAXIT" --epochs 10 --out "${OUT[$t]}" \
    > "$TMP/teach_${t}_g${g}_s${s}.log" 2>&1 &
  LAST_PID=$!
}

echo "[gen_teachers] start $(date '+%m-%d %H:%M') gpus=[$GPUS] target=$TARGET/teacher maxit=$MAXIT effort=high" >> "$PROG"
for g in $GPUS; do pick; t="$PICKED"; [ -z "$t" ] && break; launch "$g" "$t"; PID[$g]=$LAST_PID; SLOTT[$g]=$t; done

tick=0; END=$(( $(date +%s) + 1209600 ))  # 14-day cap
while [ "$(date +%s)" -lt "$END" ]; do
  sleep 120; tick=$((tick+1))
  done=1; for t in $TEACHERS; do [ "$(count "$t")" -ge "$TARGET" ] || done=0; done
  if [ "$done" = 1 ]; then echo "[gen_teachers] ALL DONE $(date '+%m-%d %H:%M')" >> "$PROG"; break; fi
  for g in $GPUS; do
    if ! kill -0 "${PID[$g]:-0}" 2>/dev/null; then
      pick; t="$PICKED"; if [ -n "$t" ]; then launch "$g" "$t"; PID[$g]=$LAST_PID; SLOTT[$g]=$t; fi
    fi
  done
  if [ $((tick % 5)) -eq 0 ]; then
    echo "$(date '+%m-%d %H:%M') luna=$(count luna) sol=$(count sol) ds=$(count ds) /$TARGET running=$(pgrep -f gen_bo_traj.py | grep -cve '^$')" >> "$PROG"
  fi
done
echo "[gen_teachers] exit $(date '+%m-%d %H:%M') luna=$(count luna) sol=$(count sol) ds=$(count ds)" >> "$PROG"
