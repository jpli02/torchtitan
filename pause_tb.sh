#!/bin/bash
# Pause the Terminal-Bench experiment batch (user request 2026-10-29):
#  - comp_queue.sh waiter (nothing started yet)           -> kill
#  - v3 replicate training (step 10050, ckpt step-10000)   -> kill launcher then trainer; resumable from step-10000
#  - take-turns 4-epoch launcher                          -> kill the launcher only; the epoch-2 eval that is 16/22
#    done keeps running under eval_gated.sh, which stops its own server when the tb run ends
set -u
T=/home/jli199/.claude/jobs/c0d2da0a/tmp
echo "=== PAUSE $(date '+%m-%d %H:%M') ===" | tee -a "$T/pause_tb.txt"
kill 3249717 2>/dev/null && echo "killed comp_queue.sh (3249717)" | tee -a "$T/pause_tb.txt"
kill 3166858 2>/dev/null && echo "killed train_v3_rep.sh launcher (3166858)" | tee -a "$T/pause_tb.txt"
sleep 1
kill 1183580 1183991 2>/dev/null && echo "killed v3 replicate torchrun+trainer (1183580 1183991) at step ~10050; resume from checkpoint/step-10000" | tee -a "$T/pause_tb.txt"
kill 3166157 2>/dev/null && echo "killed train_tvary_r1_4ep.sh launcher (3166157); epoch-2 eval (tb run 1509604, server 1505512) left to finish" | tee -a "$T/pause_tb.txt"
sleep 8
echo "still alive: $(ps -o pid= -p 3249717,3166858,1183580,1183991,3166157 2>/dev/null | tr '\n' ' ')" | tee -a "$T/pause_tb.txt"
echo "ep2 eval alive: tb=$(ps -o pid= -p 1509604 2>/dev/null | wc -l) server=$(ps -o pid= -p 1505512 2>/dev/null | wc -l)" | tee -a "$T/pause_tb.txt"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -i 3,5 | tr '\n' ';' | tee -a "$T/pause_tb.txt"; echo
