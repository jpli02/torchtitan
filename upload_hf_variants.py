import os
for line in open(os.path.expanduser("~/.boptim_keys.env")):
    line = line.strip()
    if line.startswith("export "): line = line[7:]
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
who = api.whoami()["name"]
repo = f"{who}/terminal-bench-variant-trajectories"
GEN = "/home/jli199/boptim_scratch/gen_out"
f = f"{GEN}/variant_trajectories.jsonl"
n = sum(1 for _ in open(f))
readme = f"""---
license: cc-by-4.0
tags: [terminal-bench, agent-trajectories, rejection-sampling, deepseek, sft, distillation, test-similar]
---

# Terminal-Bench test-similar variant trajectories ({n})

Verified, **near-distribution** SFT trajectories for Terminal-Bench, distilled from LLM-synthesized task VARIANTS of the 80 TB eval tasks (similar skill/difficulty, **changed scenario** — not copies, uncontaminated).

Pipeline:
1. **Task synthesis:** gpt-6-luna (high reasoning effort) generates a similar-but-changed variant of each of the 80 TB tasks (instruction + Dockerfile + files + pytest + reference solution).
2. **Validation:** each variant is built and kept only if its tests FAIL on the fresh container and PASS after the reference solution — **139 valid tasks across 58/80 seeds**.
3. **Distillation:** deepseek-v4.1-flash runs as the terminus-2 agent (pass@4); only `is_resolved` (all tests pass) trajectories kept.

- **{n} trajectories**, all unique, terminus-2 `messages` format (assistant-only loss), median 6 assistant turns.
- File: `variant_trajectories.jsonl`.

Companion to the off-distribution set `terminal-bench-deepseek-flash-300` (Nemotron-sourced). This is the "test-similar, verified" arm of a distribution-matching distillation study.
"""
open("/tmp/HF_VAR_README.md", "w").write(readme)
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=f, path_in_repo="variant_trajectories.jsonl", repo_id=repo, repo_type="dataset")
api.upload_file(path_or_fileobj="/tmp/HF_VAR_README.md", path_in_repo="README.md", repo_id=repo, repo_type="dataset")
print(f"UPLOADED to https://huggingface.co/datasets/{repo} (private)  rows={n}")
