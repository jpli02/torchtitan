#!/bin/bash
# GPU 1 queue of 2026-11-21: after the gpt-6-luna LiveCodeBench threshold search (run_experiment, PID given as LUNA_PID) ends,
#   1. the released gate at q = 0.3 on the 342 problems with the exact loop counter (the gate-NAS baseline in its own units;
#      earlier runs of this router predate the counter), three workers on card 1;
#   2. the LLM-agent arm of the gate NAS (gpt-6-luna through the OpenAI API, same objective and budget as the GP arm on card 0).
set -u
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft || exit 1
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
LUNA_PID=${LUNA_PID:?set LUNA_PID}
until ! kill -0 "$LUNA_PID" 2>/dev/null; do sleep 60; done
echo "luna search ended $(date '+%m-%d %H:%M')"
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | head -n 1)" -ge 37000 ]; do sleep 60; done
/home/jli199/boptim-agent/.venv/bin/python -I randrouter/lcb_real_eval2.py --ckpt /home/jli199/boptim_scratch/ouro_base_kvshare --thr 0.3,0.3,0.3 \
  --tag base_gate0.3_exec --cards 1:3 --chunk 12 --fallback 1 > "$TMP/real_gate03_exec.log" 2>&1
echo "gate 0.3 reference exited $? $(date '+%m-%d %H:%M')"
OPTIM=luna GPU=1 PROVIDER=openai ITERS=25 SEED=1 bash boptim_lcb/launch_gate_nas.sh
echo "GPU1 QUEUE done $(date '+%m-%d %H:%M')"
