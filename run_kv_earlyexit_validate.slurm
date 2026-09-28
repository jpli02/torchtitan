#!/bin/bash
#SBATCH --job-name=kv_ee_val
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=00:40:00
#SBATCH --output=logs/kv_ee_val_%j.out
#SBATCH --error=logs/kv_ee_val_%j.err

# Validate KV-cache + early-exit against the trusted no-cache adaptive decoder:
# same EvalPlus-canonical prompt, no-system, early_exit_threshold=0.9, same 16
# problems, same base checkpoint. Compare BOTH pass@1 and avg_loops.
#   GPU0 = no-cache adaptive (reference)   GPU1 = KV-cache + early exit (new)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1

HF=./assets/hf/Ouro-1.4B
IDS=outputs/softcheck_task_ids.txt          # 16 problems
OUT=outputs/kv_ee_val_${SLURM_JOB_ID}; mkdir -p "$OUT"
COMMON="--module ouro --config ouro_1_4b --hf_checkpoint $HF --max_gen_toks 512 \
        --early_exit_threshold 0.9 --task_ids_file $IDS --evalplus_prompt \
        --no_system --evalplus_score"

# GPU0: no-cache adaptive reference
( CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v python -u scripts/evaluate_humaneval_evalplus.py \
      $COMMON --output_path "$OUT/nocache.jsonl" \
      --eval_summary_json "$OUT/nocache.summary.json" > "$OUT/nocache.log" 2>&1 ) &
# GPU1: KV-cache + early exit (new)
( CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v python -u scripts/evaluate_humaneval_evalplus.py \
      $COMMON --kv_cache --output_path "$OUT/kv.jsonl" \
      --eval_summary_json "$OUT/kv.summary.json" > "$OUT/kv.log" 2>&1 ) &
wait

python - "$OUT" <<'PY' | tee "$OUT/compare.txt"
import json, sys
OUT = sys.argv[1]
def summ(tag):
    d = json.load(open(f"{OUT}/{tag}.summary.json"))["ouro_eval_summary"]
    return d
nc, kv = summ("nocache"), summ("kv")
print(f"{'metric':<16}{'no-cache':>14}{'kv+earlyexit':>16}")
for k in ("pass@1", "avg_loops", "loop_count", "n_problems"):
    a, b = nc.get(k), kv.get(k)
    fa = f"{a:.4f}" if isinstance(a, float) else str(a)
    fb = f"{b:.4f}" if isinstance(b, float) else str(b)
    print(f"{k:<16}{fa:>14}{fb:>16}")
# verdict
dp = abs((nc.get('pass@1') or 0) - (kv.get('pass@1') or 0))
dl = abs((nc.get('avg_loops') or 0) - (kv.get('avg_loops') or 0))
print(f"\n|d pass@1| = {dp:.4f}   |d avg_loops| = {dl:.4f}")
print("VERDICT:", "MATCH (within bf16 noise)" if dp <= 0.07 and dl <= 0.3
      else "DIVERGENT -- investigate")
PY

echo "=== timing (wall) ==="
grep -hE "Wrote .* in |Elapsed \(wall" "$OUT"/nocache.log "$OUT"/kv.log 2>/dev/null
echo "Artifacts: $OUT"
