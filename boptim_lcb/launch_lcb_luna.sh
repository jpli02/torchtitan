#!/bin/bash
# LLM-as-optimizer (through OpenRouter) on the pooled LiveCodeBench router objective: all 342 problems, 10 new evaluations
# after the same 7 warm-start records the Gaussian-process run had (user request 2026-11-20).
#   default run: openai/gpt-6-luna ("try gpt-6-luna", 2026-11-20), config ouro_lcb_router_pooled_luna6.yaml, work dir ..._luna6
#   the first run of that day used openai/gpt-5.6-luna (RUN=luna LUNA_MODEL=openai/gpt-5.6-luna); it was stopped during its
#   first evaluation (proposal (0.25, 0.3, 0.3): pass@1 15.2 % at 2.07 executed loops per token).
# One evaluation at a time on one card: generation (about 3 hours next to a neighbour's jobs) + one speed calibration of the
# candidate and one of full depth, back to back (LCB_CAL_ATTEMPTS=1: another user computes on GPU 1 all day, so the three
# attempts of the default only repeat a contaminated measurement; the 5.6 run spent 4.5 hours on them).
#   GPU=1 bash boptim_lcb/launch_lcb_luna.sh        start detached with setsid nohup; log: <job tmp>/lcb_<RUN>.log
# The API key is read from ~/.boptim_keys.env and never echoed.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
RUN=${RUN:-luna6}
export LUNA_MODEL=${LUNA_MODEL:-openai/gpt-6-luna}
export LUNA_WORK=/home/jli199/boptim_scratch/bo_lcb_router_pooled_${RUN}
CFG=ouro_lcb_router_pooled_${RUN}.yaml
GPU=${GPU:-1}
[ -f "$WT/boptim_lcb/$CFG" ] || { echo "no config $CFG"; exit 1; }
grep -q "chatgpt: \"$LUNA_MODEL\"" "$WT/boptim_lcb/$CFG" && grep -q "ouro_lcb_work_dir: $LUNA_WORK\$" "$WT/boptim_lcb/$CFG" \
  || { echo "config $CFG does not name model $LUNA_MODEL and work dir $LUNA_WORK"; exit 1; }
set -a
. "$HOME/.boptim_keys.env"
set +a
# PROVIDER=openrouter (default): model ids like openai/gpt-6-luna through OpenRouter with OPENROUTER_API_KEY.
# PROVIDER=openai: OpenAI's own API with OPENAI_API_KEY from the key file (user's key of 2026-11-21); model ids without the
# "openai/" prefix (gpt-6-luna, gpt-6-sol, ...). The config's chatgpt model must match LUNA_MODEL either way.
if [ "${PROVIDER:-openrouter}" = openai ]; then
  [ -n "${OPENAI_API_KEY:-}" ] || { echo "no OPENAI_API_KEY in ~/.boptim_keys.env"; exit 1; }
  case "$LUNA_MODEL" in openai/*) echo "PROVIDER=openai needs a model id without the openai/ prefix (got $LUNA_MODEL)"; exit 1;; esac
  export OPENAI_BASE_URL=https://api.openai.com/v1 GPT_MAX_TOKENS=16000
else
  KEY=${OPENROUTER_API_KEY:-${QWEN_API_KEY:-}}
  [ -n "$KEY" ] || { echo "no OpenRouter key in ~/.boptim_keys.env"; exit 1; }
  export OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY="$KEY" GPT_MAX_TOKENS=16000
fi
export BOPTIM_PROMPT_TEMPLATE=ouro_lcb_router_prompt.txt
export LCB_CAL_ATTEMPTS=${LCB_CAL_ATTEMPTS:-1}
cd /home/jli199/boptim-agent || exit 1
cp "$WT/boptim_lcb/ouro_lcb_router_prompt.txt" optim/ouro_lcb_router_prompt.txt
cp "$WT/boptim_lcb/$CFG" "configs/$CFG"
cp "$WT/boptim_lcb/ouro_lcb_router.py" objective/ouro_lcb_router.py
mkdir -p "$LUNA_WORK"
: > "$LUNA_WORK/gpus.txt"
./.venv/bin/python "$WT/boptim_lcb/make_warm_start_luna.py" 2>&1 | grep -v Warning
echo "LLM search ($LUNA_MODEL) starts on gpu $GPU $(date '+%m-%d %H:%M')"
LCB_BO_GPUS=$GPU ./.venv/bin/python -u run_experiment.py --config "configs/$CFG" --log-prefix "lcb_router_pooled_${RUN}" \
  > "$TMP/lcb_${RUN}.log" 2>&1
echo "LUNA search exited $? $(date '+%m-%d %H:%M')"
