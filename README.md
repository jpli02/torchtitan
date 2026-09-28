# Ouro on torchtitan

Terminal-Bench agent fine-tuning for [ByteDance Ouro](https://huggingface.co/ByteDance/Ouro-1.4B), a looped
("universal") Transformer that re-applies its layers up to 4 times per token. Built on
[torchtitan](https://github.com/pytorch/torchtitan); this repository adds the Ouro model and configs, a 2.6B
flavor, an 8-bit optimizer path, and the scripts that build terminus-2 SFT data, train, serve and evaluate on
Terminal-Bench. Every number and setting is on the
[boptim Experiments page](https://claude.ai/artifact/RSGyFVwqfHdoFTnoBGNKib); the scripts are indexed in
[`research/README.md`](research/README.md). This repo is the `torchtitan/` submodule of
[boptim-agent](https://github.com/research4pan/boptim-agent).

## Model flavors and configs

Model code: [`torchtitan/models/ouro/`](torchtitan/models/ouro). HF assets go under `assets/hf/<model>`
(`scripts/download_hf_assets.py`).

| Flavor | Layers | Notes |
|---|---|---|
| `debugmodel` | 6 | smoke tests only |
| `1.4B` | 24 | Ouro-1.4B / Ouro-1.4B-Thinking |
| `2.6B` | 48 | the 1.4B backbone at 2x depth, same recipe |

| Config | What it is for | Key environment knobs |
|---|---|---|
| `ouro_debugmodel` | smoke test | — |
| `ouro_1_4b` | pre-training / continued training | — |
| `ouro_1_4b_thinking_terminal_sft` | full-model SFT as a terminus-2 agent: assistant-only loss, Stage-I loss with 0.01 entropy bonus, seq 4096, 1 GPU | `OURO_INIT_FROM`, `OURO_SFT_MIX`, `OURO_SFT_LOCAL_JSONL`, `OURO_SFT_MIN_CMDS`, `OURO_SFT_REQUIRE_COMPLETE` |
| `ouro_2_6b_thinking_terminal_sft` | the same recipe on 2.6B with `optimizer.name = "AdamW8bit"` | as above |

**AdamW8bit** (`torchtitan/components/optimizer.py`, from `torchao.optim`) keeps optimizer states in int8:
2.6B full fine-tuning with fp32 AdamW needs ~21 GB of moments and does not fit one 46 GB card, and FSDP
checkpoint loading hangs for this model. With 8-bit states a 2.6B run peaks at 33.9 GB on a single GPU.

## Quick start

```bash
pip install -r requirements.txt            # plus torchao for AdamW8bit
MODULE=ouro CONFIG=ouro_debugmodel NGPU=1 ./run_train.sh --training.steps=20                      # smoke
OURO_SFT_LOCAL_JSONL=/path/to/train.jsonl MODULE=ouro CONFIG=ouro_1_4b_thinking_terminal_sft NGPU=1 \
    ./run_train.sh --training.steps=20000                                                          # agent SFT
```

The research drivers call `torchtitan.train` directly rather than `run_train.sh`, so the wrapper's default
batch-size arguments do not override the config; `research/terminal_sft/train/run_v3.sh` is the canonical
invocation.

## Terminal-Bench pipeline

| Stage | Scripts (`research/terminal_sft/`) |
|---|---|
| Data | `data/build_tb80_sft.py` (verified TB-2 leaderboard trajectories + domain-reweighted TerminalTraj), `data/build_tb80sft_v3.py` + `data/skill_mine.py` (thicken thin skills), `data/build_skill_sft.py` (one skill at ~40 %), `data/build_oracle12.py` (train-on-test diagnostic), coverage and contamination audits |
| Train | `train/run_v3.sh` (1.4B, 20k steps), `train/run_26b.sh` (2.6B), `train/run_skill.sh` / `train/run_skill_26b.sh` + `train/run_all_skills_26b.sh` (per-skill probes) |
| Serve + eval | `eval/eval_parallel.sh` (12-task subset sharded over free GPUs), `eval/eval_gated.sh` + `eval/verify_agent.py` (refuses a `task_complete` that ran no commands), `eval/eval80_26b.sh` (full 80 tasks), `eval/tally12.py`, `eval/tally80.py` |
| Teacher generation | `gen/convert_harbor.py` (Nemotron synthetic tasks to TB format), `gen/gen_traj.sh` / `gen/gen_all.sh` (a strong API model as the agent, rejection-sampled by each task's tests), `gen/harvest_traj.py` |

The benchmark harness (`terminal-bench 0.2.18`, local `tb_tasks/` copy of terminal-bench-core 0.1.1) is a
separate checkout at `~/terminal_bench_eval`.

### Running an experiment

One driver does the whole chain: train, export a serving dir, evaluate. Example, the tb80sft v3 recipe for
Ouro-1.4B-Thinking: 20k SFT steps on one GPU (lr 2e-5, seq 4096, about 3.3 s/step, roughly 18 h), then the
12-task subset evaluated twice per task with one Ouro server per free GPU.

```bash
bash research/terminal_sft/train/run_v3.sh          # STEPS=... to shorten
```

Results land in `~/.claude/jobs/c0d2da0a/tmp/tb80sft_v3_chain.txt` (training tail, then per-task resolved /
unresolved), per-trial JSON under `/tmp/tb_runs/<run>/`. To evaluate an existing checkpoint, or to use the gated
agent:

```bash
CKPT=/path/to/serving_dir BASENAME=my_ckpt GATE=1 bash research/terminal_sft/eval/eval_parallel.sh
```

`run_26b.sh` is the same chain for Ouro-2.6B-Thinking, `run_skill.sh` / `run_skill_26b.sh` the per-skill
probes, `eval80_26b.sh` the full 80-task protocol.

## Results at a glance

| Experiment | Result |
|---|---|
| Ouro-1.4B-Thinking | pretrained 0/80 and 0/24; every SFT recipe (1k to 20k steps, five data mixes) lands at 3-5/24 and 2-4/80 |
| Per-skill data matching (19 skills at 1.4B, 10 at 2.6B) | no task solved on both attempts; a handful of 1/2 flips at the noise floor |
| Ouro-2.6B-Thinking, same data and recipe | 6/24 ungated, 8/24 gated, 6/80; first checkpoint to solve `crack-7z-hash.easy` |
| Teacher generation (deepseek, pass@4, all-tests-pass filter) | 44 verified on-distribution trajectories after dedup; the 2.6B run on them is next |

Model capacity moved the benchmark; data curation did not.

## Practical notes

- Single GPU only: FSDP checkpoint load hangs for the 2.6B flavor, and seq-4096 runs OOM when a co-tenant holds
  ~15 GB of the card. `run_skill*.sh` pick the freest card and accept `SEQLEN=2048`.
- `WANDB_MODE=disabled` (or online) for long runs; offline mode captured stdout and froze the log once.
- A torchtitan checkpoint saves weights only. Export with `scripts/export_dcp_to_hf.py` and copy `config.json`,
  tokenizer and modeling files from the base assets into the serving dir, or the eval server will not load it.
- Earlier work on Bayesian optimisation of the exit gate and a router NAS lives in `research/bo_gate/` and
  `research/qwen_bo_agent/`, driven from boptim-agent; see the experiments page appendix.

## Slurm

`research/pretrain/run_ouro_train.slurm` wraps `run_train.sh`. Edit its `#SBATCH` lines and the venv activation
for your site, then `sbatch research/pretrain/run_ouro_train.slurm` (`CONFIG`, `NGPU` and trailing
`--training.*` overrides pass through).

## Installation

```bash
git clone --recurse-submodules git@github.com:research4pan/boptim-agent.git && cd boptim-agent/torchtitan
pip install -r requirements.txt      # a recent PyTorch nightly, plus torchao for AdamW8bit
```

## License

Source code is made available under a [BSD 3 license](./LICENSE); model weights and datasets referenced here
carry their own terms.
