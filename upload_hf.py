import os, json
for line in open(os.path.expanduser("~/.boptim_keys.env")):
    line = line.strip()
    if line.startswith("export "): line = line[7:]
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from huggingface_hub import HfApi
tok = os.environ["HF_TOKEN"]
api = HfApi(token=tok)
who = api.whoami()["name"]
repo = f"{who}/terminal-bench-deepseek-flash-300"
GEN = "/home/jli199/boptim_scratch/gen_out"
nflash = sum(1 for _ in open(f"{GEN}/deepseek_flash_generated.jsonl"))
ndeep = sum(1 for _ in open(f"{GEN}/deepseek_generated.jsonl"))
readme = f"""---
license: cc-by-4.0
task_categories: [other]
tags: [terminal-bench, agent-trajectories, rejection-sampling, deepseek, sft]
---

# Terminal-Bench deepseek-v4.1-flash trajectories ({nflash})

Rejection-sampled **successful** terminus-2 agent trajectories for Terminal-Bench SFT.

- **Teacher:** deepseek-v4.1-flash (reasoning), run as the terminus-2 agent via litellm/OpenRouter.
- **Tasks:** NVIDIA `Nemotron-Terminal-Synthetic-Tasks` (CC-BY-4.0) converted to Terminal-Bench 0.2.18; **disjoint from the 80 TB eval tasks**.
- **Gate:** each attempt graded by the task's own pytest; only `is_resolved` (all tests pass) trajectories kept (pass@k sampling).
- **Format:** one JSON object per line, `{{"messages": [{{"role": "user"/"assistant", "content": ...}}]}}`, assistant-only loss. Median ~4 assistant turns.

## Files
- `deepseek_flash_generated.jsonl` — {nflash} deepseek-v4.1-flash trajectories (primary).
- `deepseek_generated.jsonl` — {ndeep} all-deepseek (flash + a few deepseek-chat-v3-0324).

Pipeline: `convert_harbor.py` -> `gen_traj.sh` -> `harvest_traj.py`.
"""
open("/tmp/HF_README.md", "w").write(readme)
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for fn in ["deepseek_flash_generated.jsonl", "deepseek_generated.jsonl"]:
    api.upload_file(path_or_fileobj=f"{GEN}/{fn}", path_in_repo=fn, repo_id=repo, repo_type="dataset")
api.upload_file(path_or_fileobj="/tmp/HF_README.md", path_in_repo="README.md", repo_id=repo, repo_type="dataset")
print(f"UPLOADED to https://huggingface.co/datasets/{repo} (private)  flash={nflash} deepseek={ndeep}")
