#!/bin/bash
#SBATCH --job-name=kv_sft_val
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=01:00:00
#SBATCH --output=logs/kv_sft_val_%j.out
#SBATCH --error=logs/kv_sft_val_%j.err

# Validate the DCP->HF bridge for the KV-cache boptim eval:
#  1) SFT the exit gate twice with the SAME seed -> identical weights, two formats:
#       CKPT_DCP (default DCP)   and   CKPT_HF (--checkpoint.last_save_in_hf)
#  2) assemble CKPT_HF into a full HF model dir (base config/modeling/tokenizer +
#     the SFT safetensors)
#  3) eval the DCP checkpoint via the no-cache adaptive decoder (reference) and the
#     HF checkpoint via KV-cache + early exit, on the same 16 problems, threshold 0.9
#  4) compare pass@1 AND avg_loops -> confirms SFT weights load correctly into the
#     KV path and the trained gate's early exit matches the reference.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1
export OURO_ADAPTIVE_GAMMA=0.1   # a gamma that trains the gate to exit (like boptim)

BASE=./assets/hf/Ouro-1.4B
IDS=outputs/softcheck_task_ids.txt
OUT=outputs/kv_sft_val_${SLURM_JOB_ID}; mkdir -p "$OUT"
STEPS=100; SEED=42; NG=2

train () {  # $1=ckpt_folder  $2..=extra args
    local folder=$1; shift
    python -m torch.distributed.run --nproc_per_node=$NG \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
        -m torchtitan.train --module ouro --config ouro_1_4b_sft \
        --optimizer.lr 3e-4 --dump_folder "$OUT/dump" \
        --checkpoint.folder "$folder" --checkpoint.interval $STEPS \
        --training.steps $STEPS --training.local_batch_size 2 \
        --parallelism.data_parallel_shard_degree $NG --debug.seed $SEED "$@"
}

echo "### 1) SFT (DCP) ###"
train "$OUT/ckpt_dcp" > "$OUT/train_dcp.log" 2>&1
echo "### 2) SFT (HF safetensors, same seed) ###"
train "$OUT/ckpt_hf" --checkpoint.last_save_in_hf --checkpoint.last_save_model_only \
    > "$OUT/train_hf.log" 2>&1

# locate the two step dirs
DCP=$(ls -d "$OUT"/ckpt_dcp/step-* 2>/dev/null | tail -1)
HF=$(ls -d "$OUT"/ckpt_hf/step-* 2>/dev/null | tail -1)
echo "DCP ckpt: $DCP"; echo "HF ckpt: $HF"
echo "HF ckpt contents:"; ls -la "$HF"

# 3) assemble HF ckpt into a full model dir: symlink base config/modeling/tokenizer
for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json \
         tokenizer_config.json special_tokens_map.json vocab.json merges.txt \
         generation_config.json; do
    [ -e "$BASE/$f" ] && [ ! -e "$HF/$f" ] && ln -s "$(readlink -f "$BASE/$f")" "$HF/$f"
done
echo "HF ckpt after assembling:"; ls "$HF"

COMMON="--module ouro --config ouro_1_4b --max_gen_toks 512 --early_exit_threshold 0.9 \
        --task_ids_file $IDS --evalplus_prompt --no_system --evalplus_score"

echo "### 4) eval: no-cache(DCP) [GPU0] || KV(HF) [GPU1] ###"
( CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_humaneval_evalplus.py $COMMON \
    --checkpoint "$DCP" --output_path "$OUT/nocache.jsonl" \
    --eval_summary_json "$OUT/nocache.summary.json" > "$OUT/eval_nocache.log" 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 python -u scripts/evaluate_humaneval_evalplus.py $COMMON \
    --kv_cache --hf_checkpoint "$HF" --output_path "$OUT/kv.jsonl" \
    --eval_summary_json "$OUT/kv.summary.json" > "$OUT/eval_kv.log" 2>&1 ) &
wait

python - "$OUT" <<'PY' | tee "$OUT/compare.txt"
import json, sys
OUT = sys.argv[1]
def s(tag): return json.load(open(f"{OUT}/{tag}.summary.json"))["ouro_eval_summary"]
nc, kv = s("nocache"), s("kv")
print(f"{'metric':<14}{'no-cache(DCP)':>16}{'kv+ee(HF)':>14}")
for k in ("pass@1", "avg_loops", "loop_count", "n_problems"):
    a, b = nc.get(k), kv.get(k)
    fa = f"{a:.4f}" if isinstance(a,float) else str(a)
    fb = f"{b:.4f}" if isinstance(b,float) else str(b)
    print(f"{k:<14}{fa:>16}{fb:>14}")
dp = abs((nc.get('pass@1') or 0)-(kv.get('pass@1') or 0))
dl = abs((nc.get('avg_loops') or 0)-(kv.get('avg_loops') or 0))
print(f"\n|d pass@1|={dp:.4f}  |d avg_loops|={dl:.4f}")
ok = dp <= 0.07 and dl <= 0.3 and (kv.get('avg_loops') or 9) < 3.99
print("VERDICT:", "MATCH + early-exit engaged" if ok else
      "CHECK (divergent, or avg_loops~4 => gate didn't exit)")
PY
echo "Artifacts: $OUT"
