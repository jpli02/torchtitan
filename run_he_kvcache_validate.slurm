#!/bin/bash
#SBATCH --job-name=he_kv_val
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=01:30:00
#SBATCH --output=logs/he_kv_val_%j.out
#SBATCH --error=logs/he_kv_val_%j.err

# Validate the new --kv_cache HF path against the verified no-cache torchtitan
# path: same config (--evalplus_prompt --no_system, R=4) on the SAME 16 problems.
# GPU0 = no-cache (reference), GPU1 = kv-cache (new). Then diff pass sets, count
# identical solutions, and report the speedup. Must match before any full run.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1

HF=./assets/hf/Ouro-1.4B
IDS=outputs/softcheck_task_ids.txt          # 16 problems (mixed)
OUT=outputs/he_kv_val_${SLURM_JOB_ID}; mkdir -p "$OUT"
COMMON="--module ouro --config ouro_1_4b --hf_checkpoint $HF --max_gen_toks 1024 \
        --early_exit_threshold 1.0 --task_ids_file $IDS --evalplus_prompt --no_system"

# GPU0: no-cache reference
( CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v python -u scripts/evaluate_humaneval_evalplus.py \
      $COMMON --output_path "$OUT/nocache.jsonl" > "$OUT/nocache.log" 2>&1 ) &
# GPU1: kv-cache (new path)
( CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v python -u scripts/evaluate_humaneval_evalplus.py \
      $COMMON --kv_cache --output_path "$OUT/kvcache.jsonl" > "$OUT/kvcache.log" 2>&1 ) &
wait

for tag in nocache kvcache; do
    python -m evalplus.sanitize --samples "$OUT/$tag.jsonl" >> "$OUT/$tag.san.log" 2>&1
done

python - "$OUT" <<'PY' | tee "$OUT/compare.txt"
import json, sys, signal
from evalplus.data import get_human_eval_plus
OUT = sys.argv[1]; prob = get_human_eval_plus()
def load(p): return {json.loads(l)["task_id"]: json.loads(l)["solution"]
                     for l in open(p) if l.strip()}
nc = load(f"{OUT}/nocache-sanitized.jsonl"); kv = load(f"{OUT}/kvcache-sanitized.jsonl")
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a:(_ for _ in ()).throw(TO()))
def passed(sol, tid):
    p = prob[tid]; src = sol + "\n" + p["test"] + f"\ncheck({p['entry_point']})\n"
    signal.alarm(10)
    try: exec(compile(src,"<s>","exec"), {}); return True
    except Exception: return False
    finally: signal.alarm(0)
ids = sorted(set(nc)&set(kv), key=lambda x:int(x.split('/')[1]))
identical = sum(nc[t]==kv[t] for t in ids)
ncp = {t for t in ids if passed(nc[t],t)}; kvp = {t for t in ids if passed(kv[t],t)}
print(f"problems compared: {len(ids)}")
print(f"identical solution strings: {identical}/{len(ids)}")
print(f"no-cache passes: {len(ncp)}  {sorted(ncp,key=lambda x:int(x.split('/')[1]))}")
print(f"kv-cache passes: {len(kvp)}  {sorted(kvp,key=lambda x:int(x.split('/')[1]))}")
print(f"PASS-SET MATCH: {ncp==kvp}  (differ: {sorted(ncp^kvp)})")
PY

echo "==================== KV-CACHE VALIDATION ===================="
cat "$OUT/compare.txt" 2>/dev/null
echo "--- timing (wall) ---"
grep -H "Elapsed (wall" "$OUT"/nocache.log "$OUT"/kvcache.log 2>/dev/null || \
grep -H "Wrote .* in " "$OUT"/nocache.log "$OUT"/kvcache.log 2>/dev/null
echo "Artifacts: $OUT"
