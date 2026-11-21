#!/bin/bash
# Card queues of the gate NAS (user, 2026-11-21: Stage-II gate training searched by BO, on LiveCodeBench and BFCL; BO first on
# both benchmarks, then the LLM-agent arms). Three independent queues, chosen with the first argument:
#   bash boptim_lcb/gate_nas_queues.sh card1   after the luna threshold search (LUNA_PID) ends: BFCL GP, then BFCL LLM agent
#   bash boptim_lcb/gate_nas_queues.sh card0   after the LiveCodeBench GP search (GP_PID) ends: LiveCodeBench LLM agent
#   bash boptim_lcb/gate_nas_queues.sh ref     after the BFCL teacher job (TEACHER_PID) ends: the released gate at 0.3 on the 342
#                                              LiveCodeBench problems with the exact loop counter, cards 8 and 9 (long prompts on card 1)
set -u
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft || exit 1
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
wait_pid() { until ! kill -0 "$1" 2>/dev/null; do sleep 60; done; }
wait_free() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | head -n 1)" -ge "$2" ]; do sleep 60; done; }
case "${1:-}" in
  card1)
    wait_pid "${LUNA_PID:?}"; echo "luna search ended $(date '+%m-%d %H:%M')"
    until [ -f /home/jli199/boptim_scratch/randrouter/nas_feats_bfcl_live.pt ]; do sleep 60; done
    wait_free 1 30000
    OPTIM=gp GPU=1 ITERS=25 SEED=1 bash boptim_lcb/launch_bfcl_gate_nas.sh
    OPTIM=luna GPU=1 PROVIDER=openai ITERS=25 SEED=1 bash boptim_lcb/launch_bfcl_gate_nas.sh ;;
  card0)
    wait_pid "${GP_PID:?}"; echo "LiveCodeBench GP gate NAS ended $(date '+%m-%d %H:%M')"
    wait_free 0 37000
    OPTIM=luna GPU=0 PROVIDER=openai ITERS=25 SEED=1 bash boptim_lcb/launch_gate_nas.sh ;;
  ref)
    wait_pid "${TEACHER_PID:?}"; echo "BFCL teacher job ended $(date '+%m-%d %H:%M')"
    /home/jli199/boptim-agent/.venv/bin/python -I randrouter/lcb_real_eval2.py --ckpt /home/jli199/boptim_scratch/ouro_base_kvshare --thr 0.3,0.3,0.3 \
      --tag base_gate0.3_exec --cards 8:1:8.6,9:1:8.6 --chunk 12 --fallback 1 --fallback_free_mb 12000 > "$TMP/real_gate03_exec.log" 2>&1
    echo "gate 0.3 reference exited $? $(date '+%m-%d %H:%M')" ;;
  *) echo "queue must be card0, card1 or ref"; exit 1 ;;
esac
echo "QUEUE ${1} done $(date '+%m-%d %H:%M')"
