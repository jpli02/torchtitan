#!/bin/bash
# Tail of the LiveCodeBench direct-mode chain on a second GPU (the GPU 1 chain does K=4 -> oracle -> K=2 and is
# then stopped by stop_lcb_chain_after_k2.sh): gate q=0.5, gate q=0.3, per-step (0.3,0.5,0.5), K=1.
set -u
export NAME=lcb342 DATA=/home/jli199/.claude/jobs/c0d2da0a/tmp/lcb_all.jsonl GPU=${GPU:-3}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
cd "$WT" || exit 1
bash lcb_point.sh q0.5 --steps 4 --threshold 0.5
bash lcb_point.sh q0.3 --steps 4 --threshold 0.3
bash lcb_point.sh steps0.3-0.5-0.5 --steps 4 --threshold_steps 0.3,0.5,0.5
bash lcb_point.sh k1 --steps 1
echo "=== LCB tail (gpu $GPU) done $(date '+%m-%d %H:%M') ===" | tee -a /home/jli199/.claude/jobs/c0d2da0a/tmp/bench_lcb342.txt
