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
echo "dory suite on GPU $GPU start $(date '+%m-%d %H:%M')"
NAME=gpqa_think TASK=mc DATA=$TMP/gpqa_diamond.jsonl N=198 BATCH=8 MAXNEW=2048 GPU=$GPU bash bench_pipeline.sh
GPU=$GPU VARIANT=think bash ifb_points.sh
GPU=$GPU VARIANT=direct bash ifb_points.sh
echo "dory suite done $(date '+%m-%d %H:%M')"
grep -h 'RESULT\|GRADED' "$TMP/bench_gpqa_think.txt" "$TMP/bench_ifbench_think.txt" "$TMP/bench_ifbench_direct.txt" | sed 's/"ckpt": "[^"]*", //' | cut -c1-220
