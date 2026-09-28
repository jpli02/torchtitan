# research/ — Ouro experiment scripts

Everything here was written for one fork-specific research programme: tune and train
[ByteDance Ouro](https://huggingface.co/ByteDance/Ouro-1.4B) (a looped Transformer with an early-exit gate)
and measure it on HumanEval, MBPP and Terminal-Bench. The write-up with every number is the
[boptim Experiments page](https://claude.ai/artifact/RSGyFVwqfHdoFTnoBGNKib); the top-level
[README](../README.md) explains
the model flavors, configs and pipelines.

Conventions: shell drivers set `WT=<repo root>` and `cd "$WT"` before calling `torchtitan.train`, so run them
from anywhere; sibling scripts are referenced by their full `research/...` path. Python helpers that import a
sibling put their own directory on `sys.path`. Model assets live under `<repo>/assets/hf/`, the Terminal-Bench
harness under `~/terminal_bench_eval`, run outputs under `~/boptim_scratch` and `~/.claude/jobs/c0d2da0a/tmp`.

| Directory | What is in it |
|---|---|
| `bo_gate/` | Exit-gate Bayesian optimisation and its evaluation protocol: HumanEval / MBPP EvalPlus runs (local `run_he_*.sh`, Slurm `run_ouro_*.slurm`), KV-cache and parity validation, the 6-D router NAS (`run_nas.sh`, `modeling_ouro_nas.py`, `nas_compare.py`), `random_search.py` baseline, `.env.vllm` for the vLLM eval venv. The BO loop itself is in `boptim-agent`. |
| `pretrain/` | `run_ouro_train.slurm`, the original Slurm wrapper around `run_train.sh`. |
| `terminal_sft/data/` | Building terminus-2 SFT sets: `build_tb80_sft.py` (verified TB-2 trajectories + domain-reweighted TerminalTraj), `build_tb80sft_v3.py` + `skill_mine.py` (thin-skill thickening), `build_skill_sft.py` (one skill at ~40 %), `build_oracle*.py` (train-on-test diagnostics), coverage / contamination / distribution audits, `generated_data/` (first harvested batch). |
| `terminal_sft/train/` | Training drivers: `run_v3.sh` (1.4B, 20k steps), `run_26b.sh` (2.6B, AdamW8bit), `run_skill.sh` / `run_skill_26b.sh` + `run_all_skills_26b.sh` (per-skill probes), earlier recipes (`run_tb80_sft.sh`, `run_traj*.sh`, `run_batching_sft.sh`, `run_oracle*.sh`), checkpoint export (`convert_sweep.sh`). |
| `terminal_sft/eval/` | Serving and Terminal-Bench evaluation: `eval_gated.sh` (one server, N tasks, optional `verify_agent.py` gate), `eval_parallel.sh` (12 tasks sharded over free GPUs), `eval80_26b.sh` (full 80), tallies (`tally12.py`, `tally80.py`, `partial_credit.py`, `final_skill_tally.py`), failure analyses (`headroom.py`, `premature.py`, `pushbacks.py`, `sweep_analyze.py`), `safe_cleaner.sh` for `/tmp/tb_runs`. |
| `terminal_sft/gen/` | Teacher trajectory generation: `convert_harbor.py` (Nemotron tasks to TB format), `gen_traj.sh` / `gen_all.sh` (a strong API model as the agent, rejection-sampled by each task's tests), `harvest_traj.py`, plus API and timeout probes. |
| `qwen_bo_agent/` | Distilling a BO optimiser into Qwen3-1.7B: `qwen_bo_server.py`, `bo_replay_format.py`, `gen_bo_traj.py`, `rationalize.py`, `build_bo_sft.sh`, `run_bo_sft.sh`, `bo_compare.py`, OpenRouter probes. |
| `docs/` | `slides_ouro_sft.html` and the experiments page source (`boptim-experiments.html`, public variant via `strip_skeleton.py`). |
