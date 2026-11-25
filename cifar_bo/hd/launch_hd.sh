#!/bin/bash
# One arm of the 8-knob CIFAR comparison on a list of seeds, run one after the other, detached so a daemon restart does not
# kill it:
#   ARM=luna_default SEEDS="42 43" GPU=2 bash cifar_bo/hd/launch_hd.sh
# Arms: gp10 (GP-hedge, n_initial 10 = the budget, the user's protocol: all 10 points are skopt grid points), gp5 (GP-hedge,
# n_initial 5, then 5 GP steps), random (uniform), luna_default (gpt-6-luna, the shipped prompt: anonymous x0..x7),
# luna_informed (gpt-6-luna, prompts/cifar_hd_informed.txt: what each knob is, no suggested values).
# Every run: 10 evaluations, 10 epochs, optimizer seed = objective seed. boptim-agent's own main.py runs through
# cifar_bo/hd/run_hd.py from the trajectory checkout (trajectories saved); LLM arms call OpenAI's API with the key from
# ~/.boptim_keys.env (never printed). Log: $ROOT/logs/<arm>_s<first seed>.log
set -eu
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
BOPTIM=/home/jli199/boptim-agent/.claude/worktrees/cifar-save-trajectory
PY=/home/jli199/boptim-agent/.venv/bin/python
# OBJ=cifar_hd (8 training knobs, default) or cifar_arch (the same 8 + base width, blocks per stage, kernel size)
OBJ=${OBJ:-cifar_hd}
case "$OBJ" in
  cifar_hd) ROOT=${ROOT:-/home/jli199/boptim_scratch/cifar_hd_1123}; INFORMED=cifar_hd_informed.txt ;;
  cifar_arch) ROOT=${ROOT:-/home/jli199/boptim_scratch/cifar_arch_1125}; INFORMED=cifar_arch_informed.txt ;;
  *) echo "unknown objective $OBJ"; exit 1 ;;
esac
mkdir -p "$ROOT/logs"
set -- $SEEDS
LOG=$ROOT/logs/${ARM}_s$1.log

set -a
. "$HOME/.boptim_keys.env"
set +a
export OPENAI_BASE_URL=https://api.openai.com/v1 GPT_MAX_TOKENS=16000 CUDA_VISIBLE_DEVICES=$GPU
unset BOPTIM_PROMPT_TEMPLATE GPT_REASONING_EFFORT
case "$ARM" in
  gp10) OPT=(--optim gp_hedge --n_initial 10) ;;
  gp5) OPT=(--optim gp_hedge --n_initial 5) ;;
  random) OPT=(--optim random) ;;
  luna_default) OPT=(--optim chatgpt --model gpt-6-luna) ;;
  luna_informed) OPT=(--optim chatgpt --model gpt-6-luna); export BOPTIM_PROMPT_TEMPLATE=$WT/cifar_bo/prompts/$INFORMED ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

cd "$BOPTIM"
echo "start $ARM objective $OBJ seeds $SEEDS gpu $GPU $(date '+%m-%d %H:%M:%S') prompt=${BOPTIM_PROMPT_TEMPLATE:-shipped}" > "$LOG"
setsid nohup bash -c '
  for s in '"$SEEDS"'; do
    echo "=== seed $s $(date "+%m-%d %H:%M:%S")"
    "$0" -u "$1"/cifar_bo/hd/run_hd.py --objective '"$OBJ"' "${@:3}" --max_iter 10 --optimizer-seed $s --objective-seed $s \
      --n_epochs 10 --metrics validation_accuracy --trajectory-dir "$2/trajectories/'"$ARM"'" || echo "=== seed $s FAILED rc=$?"
  done
  echo "=== done $(date "+%m-%d %H:%M:%S")"
' "$PY" "$WT" "$ROOT" "${OPT[@]}" >> "$LOG" 2>&1 < /dev/null &
echo "launched $ARM seeds $SEEDS gpu $GPU pid $!"
