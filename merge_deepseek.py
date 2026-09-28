"""Merge + dedup all deepseek-generated trajectory jsonls into
deepseek_generated.jsonl, and rebuild generated_union.jsonl (deepseek + the
pre-existing gpt-5-mini archive). Prints DEEPSEEK=<n> UNION=<n>."""
import json, hashlib, glob

GEN = "/home/jli199/boptim_scratch/gen_out"
ARCH = "/home/jli199/boptim_scratch/gen_out_archive_sept11/generated_all.jsonl"

# batch-1 per-category chat-v3 files + flash + every accumulation batch
deepseek_files = [f"{GEN}/{c}.jsonl" for c in
                  ["file_operations", "data_science", "scientific_computing",
                   "security", "debugging", "data_processing", "flash_file_operations"]]
deepseek_files += sorted(glob.glob(f"{GEN}/more_*.jsonl"))

def hsh(o):
    return hashlib.md5(json.dumps(o.get("messages", o), sort_keys=True).encode()).hexdigest()

seen = set(); ds = []
for f in deepseek_files:
    try:
        lines = [l for l in open(f) if l.strip()]
    except FileNotFoundError:
        continue
    for l in lines:
        try:
            o = json.loads(l)
        except Exception:
            continue
        h = hsh(o)
        if h in seen:
            continue
        seen.add(h); ds.append(l.rstrip("\n"))

open(f"{GEN}/deepseek_generated.jsonl", "w").write("\n".join(ds) + ("\n" if ds else ""))

# union with gpt-5-mini archive (dedup)
union = list(ds)
try:
    for l in open(ARCH):
        if not l.strip():
            continue
        try:
            o = json.loads(l)
        except Exception:
            continue
        h = hsh(o)
        if h in seen:
            continue
        seen.add(h); union.append(l.rstrip("\n"))
except FileNotFoundError:
    pass
open(f"{GEN}/generated_union.jsonl", "w").write("\n".join(union) + ("\n" if union else ""))

print(f"DEEPSEEK={len(ds)} UNION={len(union)}")
