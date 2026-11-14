#!/bin/bash
# Feasibility probes for more 2025-venue benchmarks on one card, cheapest first (gather mode, loops counted):
#   1. MMLU-Redux 2.0 (NAACL 2025), direct answer letter, 1,995 questions                                 -> bench_mmluredux_direct.txt
#   2. LiveCodeBench code execution (ICLR 2025), thinking, first N rows                                   -> bench_lcbexec_think.txt
#   3. LiveCodeBench test-output prediction (ICLR 2025), thinking                                         -> bench_lcbtest_think.txt
#   4. Test of Time arithmetic (ICLR 2025), thinking                                                      -> bench_totarith_think.txt
#   5. TableBench numerical reasoning + fact checking (AAAI 2025), thinking, batch 1 (tables up to 5k tokens) -> bench_tbench_think.txt
#   GPU=0 N=100 TB=6 MMLU_POINTS="k2 q0.5 q0.3 q0.1 steps0.3-0.5-0.5 steps0.1-0.5-0.5" bash more_bench_chain.sh
# TB = batch of the thinking probes (6 on a card with 39 GB free, 2 on a 24 GB half); MMLU_POINTS="" skips the MMLU-Redux part.
set -u
: "${GPU:?}"
N=${N:-100}; TB=${TB:-2}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
CKPT=/home/jli199/boptim_scratch/ouro_base_kvshare
cd "$WT" || exit 1
think() {  # think <name> <task> <data> <max_new> <batch> <points>
  local name=$1 task=$2 data=$3 maxnew=$4 batch=$5 points=$6 rep=$TMP/bench_$1.txt extra=""
  echo "=== THINK $name (task $task, n=$N, batch $batch, max_new $maxnew, gpu $GPU) start $(date '+%m-%d %H:%M') ===" | tee -a "$rep"
  for pt in $points; do
    case $pt in k4) extra="--steps 4" ;; k2) extra="--steps 2" ;; q0.3) extra="--steps 4 --threshold 0.3" ;; steps0.3-0.5-0.5) extra="--steps 4 --threshold_steps 0.3,0.5,0.5" ;; esac
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" depth_eval.py "$CKPT" "$data" --task "$task" \
      --n "$N" --batch "$batch" --max_new "$maxnew" $extra --out "$TMP/bench_${name}_${pt}.jsonl" > "$TMP/bench_${name}_${pt}.log" 2>&1
    grep '^RESULT' "$TMP/bench_${name}_${pt}.log" | sed "s/^RESULT /RESULT $pt /" | tee -a "$rep" | sed 's/"ckpt": "[^"]*", //' | cut -c1-260
    grep -q '^RESULT' "$TMP/bench_${name}_${pt}.log" || echo "$pt FAILED: $(grep -E 'Error' "$TMP/bench_${name}_${pt}.log" | tail -1 | cut -c1-160)" | tee -a "$rep"
  done
  echo "=== THINK $name done $(date '+%m-%d %H:%M') ===" | tee -a "$rep"
}
echo "more-benchmark probes on GPU $GPU start $(date '+%m-%d %H:%M')"
MMLU_POINTS=${MMLU_POINTS-"k4 k2 k1 q0.5 q0.3 q0.1 steps0.3-0.5-0.5 steps0.1-0.5-0.5"}
if [ -n "$MMLU_POINTS" ]; then
  NAME=mmluredux_direct TASK=mc DATA=$TMP/mmlu_redux.jsonl PREFIX='\\boxed{' MAXNEW=8 N=1995 BATCH=${MB:-8} GPU=$GPU POINTS="$MMLU_POINTS" bash direct_points.sh
fi
think lcbexec_think crux "$TMP/lcb_exec.jsonl" 1024 "$TB" "${LCB_POINTS:-k4 q0.3}"
think lcbtest_think lcbt "$TMP/lcb_testout.jsonl" 1024 "$TB" "${LCB_POINTS:-k4 q0.3}"
think totarith_think tot "$TMP/tot_arith.jsonl" 1024 "$TB" "${OTHER_POINTS:-k4}"
think tbench_think tbench "$TMP/tablebench.jsonl" 1024 1 "${OTHER_POINTS:-k4}"
echo "more-benchmark probes done $(date '+%m-%d %H:%M')"
grep -h 'RESULT\|FAILED' "$TMP"/bench_mmluredux_direct.txt "$TMP"/bench_lcbexec_think.txt "$TMP"/bench_lcbtest_think.txt "$TMP"/bench_totarith_think.txt "$TMP"/bench_tbench_think.txt 2>/dev/null | sed 's/"ckpt": "[^"]*", //' | cut -c1-200
