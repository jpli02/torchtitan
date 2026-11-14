#!/bin/bash
# Feasibility probes for five more 2025-venue benchmarks on one (possibly half-held) card, cheapest first:
#   1. MMLU-Redux 2.0 (NAACL 2025), direct answer letter, 1,995 questions, 8 depth/gate points          -> bench_mmluredux_direct.txt
#   2. LiveCodeBench code execution (ICLR 2025), thinking, first N rows: K=4, K=2, gate 0.3              -> bench_lcbexec_think.txt
#   3. LiveCodeBench test-output prediction (ICLR 2025), thinking: K=4, K=2, gate 0.3                    -> bench_lcbtest_think.txt
#   4. Test of Time arithmetic (ICLR 2025), thinking: K=4, K=2, gate 0.3                                  -> bench_totarith_think.txt
#   5. TableBench numerical reasoning + fact checking (AAAI 2025), thinking: K=4, K=2, gate 0.3           -> bench_tbench_think.txt
# Thinking probes use batch 2 (a half card has 24 GB) and N rows each (default 100); gather mode, loops counted.
#   GPU=3 bash more_bench_chain.sh
set -u
: "${GPU:?}"
N=${N:-100}; TB=${TB:-2}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
cd "$WT" || exit 1
think() {  # think <name> <task> <data> <max_new> <batch>
  local name=$1 task=$2 data=$3 maxnew=$4 batch=$5 rep=$TMP/bench_$1.txt
  echo "=== THINK $name (task $task, n=$N, batch $batch, max_new $maxnew, gpu $GPU) start $(date '+%m-%d %H:%M') ===" | tee -a "$rep"
  for pt in ${THINK_POINTS:-k4 k2 q0.3}; do
    case $pt in k4) extra="--steps 4" ;; k2) extra="--steps 2" ;; q0.3) extra="--steps 4 --threshold 0.3" ;; steps0.3-0.5-0.5) extra="--steps 4 --threshold_steps 0.3,0.5,0.5" ;; esac
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$data" --task "$task" \
      --n "$N" --batch "$batch" --max_new "$maxnew" $extra --out "$TMP/bench_${name}_${pt}.jsonl" > "$TMP/bench_${name}_${pt}.log" 2>&1
    grep '^RESULT' "$TMP/bench_${name}_${pt}.log" | sed "s/^RESULT /RESULT $pt /" | tee -a "$rep" | sed 's/"ckpt": "[^"]*", //' | cut -c1-260
    grep -q '^RESULT' "$TMP/bench_${name}_${pt}.log" || echo "$pt FAILED: $(grep -E 'Error' "$TMP/bench_${name}_${pt}.log" | tail -1 | cut -c1-160)" | tee -a "$rep"
  done
  echo "=== THINK $name done $(date '+%m-%d %H:%M') ===" | tee -a "$rep"
}
echo "more-benchmark probes on GPU $GPU start $(date '+%m-%d %H:%M')"
NAME=mmluredux_direct TASK=mc DATA=$TMP/mmlu_redux.jsonl PREFIX='\\boxed{' MAXNEW=8 N=1995 BATCH=8 GPU=$GPU bash direct_points.sh
think lcbexec_think crux "$TMP/lcb_exec.jsonl" 1024 "$TB"
think lcbtest_think lcbt "$TMP/lcb_testout.jsonl" 1024 "$TB"
think totarith_think tot "$TMP/tot_arith.jsonl" 1024 "$TB"
think tbench_think tbench "$TMP/tablebench.jsonl" 1024 1
echo "more-benchmark probes done $(date '+%m-%d %H:%M')"
grep -h 'RESULT\|FAILED' "$TMP"/bench_mmluredux_direct.txt "$TMP"/bench_lcbexec_think.txt "$TMP"/bench_lcbtest_think.txt "$TMP"/bench_totarith_think.txt "$TMP"/bench_tbench_think.txt 2>/dev/null | sed 's/"ckpt": "[^"]*", //' | cut -c1-200
