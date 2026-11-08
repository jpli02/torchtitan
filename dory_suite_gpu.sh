#!/bin/bash
# The Dory1.0-4B (arXiv 2611.03998) suite pieces a 2.6B non-agentic model can run, on one GPU:
#   GPQA Diamond, thinking regime (bench_pipeline: K=4 + oracle + proxy, K=2, gate 0.3, per-step) -> $TMP/bench_gpqa_think.txt
#   IFBench, thinking and direct variants, official verifiers                                    -> $TMP/bench_ifbench_{think,direct}.txt
# (GPQA Diamond in the direct multiple-choice regime runs separately: NAME=gpqa_direct ... bash mc_direct.sh)
#   GPU=1 bash dory_suite_gpu.sh
set -u
: "${GPU:?}"
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
PART=${PART:-all}   # gpqa | ifb | all
# batch: BATCH_FULL (default 6) on a card we hold entirely (>= 40 GB free), else the given BATCH (default 4: another user half-holds
# idle cards with 24 GB). 8 is too many for 4-loop points with 2,048-token thinking (the 4x-deep KV cache ran a 48 GB card out of memory).
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
if [ "${free:-0}" -ge 40000 ]; then BATCH=${BATCH_FULL:-6}; else BATCH=${BATCH:-4}; fi
export BATCH
echo "dory suite ($PART) on GPU $GPU (free ${free} MiB, batch $BATCH) start $(date '+%m-%d %H:%M')"
if [ "$PART" != ifb ]; then
  grep -q "BENCH gpqa_think.*done" "$TMP/bench_gpqa_think.txt" 2>/dev/null || NAME=gpqa_think TASK=mc DATA=$TMP/gpqa_diamond.jsonl N=198 BATCH=${BATCH:-8} MAXNEW=2048 GPU=$GPU bash bench_pipeline.sh
fi
if [ "$PART" != gpqa ]; then
  GPU=$GPU VARIANT=think POINTS="${IFB_THINK_POINTS:-k4 k2 q0.3}" bash ifb_points.sh
  GPU=$GPU VARIANT=direct POINTS="${IFB_DIRECT_POINTS:-k4 q0.3}" bash ifb_points.sh
fi
echo "dory suite ($PART) done $(date '+%m-%d %H:%M')"
grep -h 'RESULT\|GRADED' "$TMP/bench_gpqa_think.txt" "$TMP/bench_ifbench_think.txt" "$TMP/bench_ifbench_direct.txt" | sed 's/"ckpt": "[^"]*", //' | cut -c1-220
