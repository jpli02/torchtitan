<div align="center">

# torchtitan

#### A PyTorch native platform for training generative AI models

[![8 GPU Feature Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_features.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_features.yaml?query=branch%3Amain)
[![8 GPU Model Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_models.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_models.yaml?query=branch%3Amain)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06511-b31b1b.svg)](https://arxiv.org/abs/2410.06511)
[![ICLR](https://img.shields.io/badge/ICLR-2025-violet.svg)](https://iclr.cc/virtual/2025/poster/29620)
[![forum](https://img.shields.io/badge/pytorch-forum-DE3412.svg)](https://discuss.pytorch.org/c/distributed/torchtitan/44)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](./LICENSE)
[![pip](https://img.shields.io/pypi/v/torchtitan?color=blue)](https://pypi.org/project/torchtitan/)
[![conda](https://img.shields.io/conda/vn/conda-forge/torchtitan?color=green)](https://anaconda.org/conda-forge/torchtitan)


</div>

> **This fork** adds the Ouro looped Transformer and the experiments built on it (exit-gate search, Terminal-Bench SFT, 2.6B scaling). Jump to [Ouro fork](#ouro-fork-looped-transformer-training-exit-gate-search-terminal-bench-sft). The upstream README follows.

`torchtitan` is under extensive development. To use the latest features of `torchtitan`, we recommend using the most recent PyTorch nightly.


## Latest News
- [2025/11] AMD released an [optimized fork](https://github.com/AMD-AGI/torchtitan-amd/tree/main) of `torchtitan` for AMD GPUs.
- [2025/10] We released `torchtitan` [v0.2.0](https://github.com/pytorch/torchtitan/releases).
- [2025/10] SkyPilot now supports `torchtitan`! See the tutorial [here](https://docs.skypilot.co/en/latest/examples/training/torchtitan.html).
- [2025/07] We published [instructions](/torchtitan/models/README.md) on how to add a model to `torchtitan`.
- [2025/04] Our paper was accepted by [ICLR 2025](https://iclr.cc/virtual/2025/poster/29620).
- [2024/12] GPU MODE [lecture](https://www.youtube.com/watch?v=VYWRjcUqW6w) on torchtitan.
- [2024/07] [Presentation](https://pytorch2024.sched.com/event/1fHn3) at PyTorch Conference 2024.


## Overview

`torchtitan` is a PyTorch native platform designed for **rapid experimentation and large-scale training** of generative AI models. As a minimal clean-room implementation of PyTorch native scaling techniques, `torchtitan` provides a flexible foundation for developers to build upon. With `torchtitan` [extension points](docs/extension.md), one can easily create custom extensions tailored to specific needs.

Our mission is to accelerate innovation in the field of generative AI by empowering researchers and developers to explore new modeling architectures and infrastructure techniques.

The Guiding Principles when building `torchtitan`
* Designed to be easy to understand, use and extend for different training purposes.
* Minimal changes to the model code when applying multi-dimensional parallelism.
* Bias towards a clean, minimal codebase while providing basic reusable / swappable components.

`torchtitan` has been showcasing PyTorch's latest distributed training features, via support for pretraining Llama 3.1 LLMs of various sizes.

## Contributing

We look forward to your contributions!

* To accelerate contributions to and innovations around torchtitan, we host an [`experiments`](torchtitan/experiments) folder. New ideas should start there. To contribute, follow the [`experiments guidelines`](torchtitan/experiments/README.md).
* For fixes and contributions to core, follow these [`guidelines`](CONTRIBUTING.md).

## Llama 3.1 training

### Key features available

1. Multi-dimensional composable parallelisms
   - [FSDP2](docs/fsdp.md) with per-parameter sharding
   - [Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) (including [async TP](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487))
   - [Pipeline Parallel](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
   - [Context Parallel](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082)
2. [Meta device](https://pytorch.org/docs/stable/meta.html) initialization
3. Per-op selective and full activation checkpointing
4. [Distributed checkpointing](https://discuss.pytorch.org/t/distributed-w-torchtitan-optimizing-checkpointing-efficiency-with-pytorch-dcp/211250) (including async checkpointing)
   - [Interoperable checkpoints](docs/checkpoint.md) which can be loaded directly into [`torchtune`](https://github.com/pytorch/torchtune) for fine-tuning
5. `torch.compile` support
6. [Float8](https://discuss.pytorch.org/t/distributed-w-torchtitan-enabling-float8-all-gather-in-fsdp2/209323) support ([how-to](docs/float8.md))
7. [MXFP8 training for dense and MoE models](docs/mxfp8.md) on Blackwell GPUs.
7. DDP and HSDP
8. [TorchFT](https://github.com/pytorch/torchft) integration
9. Checkpointable data-loading, with the C4 dataset pre-configured (144M entries) and support for [custom datasets](docs/datasets.md)
10. Gradient accumulation, enabled by giving an additional `--training.global_batch_size` argument on the CLI
11. Flexible learning rate scheduler (warmup-stable-decay)
12. Loss, GPU memory, throughput (tokens/sec), TFLOPs, and MFU displayed and logged via [Tensorboard or Weights & Biases](/docs/metrics.md)
13. [Debugging tools](docs/debugging.md) including CPU/GPU profiling, memory profiling, Flight Recorder, etc.
14. All options easily configured via [Python config registry](torchtitan/models/llama3/config_registry.py) with `--module` and `--config` CLI flags
15. [Helper scripts](scripts/) to
    - download tokenizers from Hugging Face
    - convert original Llama 3 checkpoints into the expected DCP format
    - estimate FSDP/HSDP memory usage without materializing the model
    - run distributed inference with Tensor Parallel

We report [performance](benchmarks/llama3_h100_202412_torchtitan.md) on up to 512 GPUs, and verify [loss converging](docs/converging.md) correctness of various techniques.

### Dive into the code

You may want to see how the model is defined or how parallelism techniques are applied. For a guided tour, see these files first:
* [torchtitan/train.py](torchtitan/train.py) - the main training loop and high-level setup code
* [torchtitan/models/llama3/model.py](torchtitan/models/llama3/model.py) - the Llama 3.1 model definition
* [torchtitan/models/llama3/parallelize.py](torchtitan/models/llama3/parallelize.py) - helpers for applying Data Parallel, Tensor Parallel, activation checkpointing, and `torch.compile` to the model
* [torchtitan/distributed/pipeline_parallel.py](torchtitan/distributed/pipeline_parallel.py) - helpers for applying Pipeline Parallel to the model
* [torchtitan/components/checkpoint.py](torchtitan/components/checkpoint.py) - utils for saving/loading distributed checkpoints
* [torchtitan/components/quantization/float8.py](torchtitan/components/quantization/float8.py) - utils for applying Float8 techniques


## Installation

One can directly run the source code, or install `torchtitan` from a nightly build, or a stable release.

### From source

This method requires the nightly build of PyTorch, or the latest PyTorch built [from source](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source).

```bash
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu
```

> **Note:** The nightly build of `torchdata` is required when using a PyTorch nightly. Install it from the nightly index as shown above.

### Nightly builds

This method requires the nightly build of PyTorch. You can replace `cu128` with another version of cuda or an AMD GPU (e.g. `rocm6.3`).

```sh
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Stable releases
One can install the latest [stable release](https://github.com/pytorch/torchtitan/releases) of `torchtitan` via `pip` or `conda`.
```sh
pip install torchtitan
```
```sh
conda install conda-forge::torchtitan
```
Note that each stable release pins the nightly versions of `torch` and `torchao`. Please see [release.md](docs/release.md) for more details.

### Downloading a tokenizer

`torchtitan` currently supports training Llama 3.1 (8B, 70B, 405B) out of the box. To get started training these models, we need to download the tokenizer. Follow the instructions on the official [meta-llama](https://huggingface.co/meta-llama/Llama-3.1-8B) repository to ensure you have access to the Llama model weights.

Once you have confirmed access, you can run the following command to download the Llama 3.1 tokenizer to your local machine.

```bash
# Get your HF token from https://huggingface.co/settings/tokens

# Llama 3.1 tokenizer
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

### Start a training run
Llama 3 8B model locally on 8 GPUs

```bash
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh
```

### Multi-Node Training
For training on ParallelCluster/Slurm type configurations, you can use the `multinode_trainer.slurm` file to submit your sbatch job.

To get started adjust the number of nodes and GPUs
```
#SBATCH --ntasks=2
#SBATCH --nodes=2
```

Then start a run where `nnodes` is your total node count, matching the sbatch node count above.

```
srun torchrun --nnodes 2
```

If your gpu count per node is not 8, adjust `--nproc_per_node` in the torchrun command and `#SBATCH --gpus-per-task` in the SBATCH command section.

## Ouro fork: looped Transformer training, exit-gate search, Terminal-Bench SFT

This fork (`github.com/jpli02/torchtitan`) adds [ByteDance Ouro](https://huggingface.co/ByteDance/Ouro-1.4B), a
looped ("universal") Transformer that re-applies its layers up to 4 times per token and learns an
early-exit gate, plus the research scripts built on it. The experiment write-up with every number
is the [boptim Experiments page](https://claude.ai/artifact/RSGyFVwqfHdoFTnoBGNKib).

### Where things live

| Branch | Contents |
|---|---|
| `worktree-ouro-terminal-sft` | Everything below: model, configs, SFT data builders, eval harness drivers, teacher generation. Research scripts sit at the repo root. |
| `ouro-boptim` | The checkout used by [`boptim-agent`](https://github.com/jpli02/boptim-agent) for the exit-gate Bayesian-optimisation objective. |
| `ouro-simple` | The original Slurm training wrapper (`run_ouro_train.slurm`); see the end of this section. |

Model code: [`torchtitan/models/ouro/`](torchtitan/models/ouro) (`model.py`, `parallelize.py`, `state_dict_adapter.py`,
`config_registry.py`). HF assets are expected under `./assets/hf/<model>` (`Ouro-1.4B`, `Ouro-1.4B-Thinking`, ...);
`scripts/download_hf_assets.py` fetches them.

### Model flavors and configs

| Flavor | Layers | Notes |
|---|---|---|
| `debugmodel` | 6 | dim 256, bundled tokenizer under `tests/assets/tokenizer`; smoke tests only |
| `1.4B` | 24 | ByteDance Ouro-1.4B / Ouro-1.4B-Thinking |
| `2.6B` | 48 | The 1.4B backbone at 2x depth; trained with the same recipe, only the optimizer-state precision changes |

| Config (`--config`) | What it is for | Key environment knobs |
|---|---|---|
| `ouro_debugmodel` | CPU/1-GPU smoke test | — |
| `ouro_1_4b` | pre-training / continued training | — |
| `ouro_1_4b_sft` | **gate-only SFT** used by the boptim exit-gate objective (backbone frozen, 2,049 trainable params, 100 steps) | `OURO_ADAPTIVE_GAMMA`, `OURO_LOSS_STAGE` (`stage1_entropy` / `stage2_adaptive`), `OURO_RANDOM_INIT_GATE` |
| `ouro_1_4b_thinking_terminal_sft` | **full-model SFT** of Ouro-1.4B-Thinking as a terminus-2 terminal agent (assistant-only loss, Stage-I loss with 0.01 entropy bonus, seq 4096, 1 GPU) | `OURO_INIT_FROM` (continue from a checkpoint), `OURO_SFT_MIX`, `OURO_SFT_LOCAL_JSONL`, `OURO_SFT_MIN_CMDS`, `OURO_SFT_REQUIRE_COMPLETE` |
| `ouro_2_6b_thinking_terminal_sft` | the same recipe on the 2.6B flavor, with `optimizer.name = "AdamW8bit"` | as above |

**AdamW8bit.** `torchtitan/components/optimizer.py` accepts `AdamW8bit` (from `torchao.optim`), which keeps
optimizer states in int8. It exists because 2.6B full fine-tuning with fp32 AdamW needs ~21 GB of moments and
does not fit one 46 GB card, and FSDP checkpoint loading hangs on a cross-rank collective for this model (both HF
and DCP paths). With 8-bit states a 2.6B run peaks at 33.9 GB on a single GPU.

### Quick start (single node, no Slurm)

```bash
pip install -r requirements.txt            # plus torchao for AdamW8bit
# smoke test
MODULE=ouro CONFIG=ouro_debugmodel NGPU=1 ./run_train.sh --training.steps=20
# terminal-agent SFT of Ouro-1.4B-Thinking on a local JSONL of terminus-2 conversations
OURO_SFT_LOCAL_JSONL=/path/to/train.jsonl MODULE=ouro CONFIG=ouro_1_4b_thinking_terminal_sft NGPU=1 ./run_train.sh --training.steps=20000
```

The research drivers below call `torchtitan.train` directly (not `run_train.sh`) so that the wrapper's default
batch-size arguments do not override the config; read `run_v3.sh` for the canonical invocation.

### Pipelines

**1. Exit-gate Bayesian optimisation (HumanEval / MBPP).** Each BO evaluation runs `ouro_1_4b_sft` for 100
steps at a proposed `adaptive_gamma`, exports the gate, and scores loops-per-token and pass@1.

| Script | Role |
|---|---|
| `scripts/evaluate_humaneval_evalplus.py`, `scripts/evaluate_mbpp.py` | benchmark evals with early-exit threshold, KV-cache and prompt-format flags |
| `scripts/export_dcp_to_hf.py`, `scripts/extract_router.py` | DCP checkpoint to HF serving dir; pull the gate out as a `.safetensors` |
| `scripts/ouro_openai_server.py` | OpenAI-compatible `/v1/chat/completions` server for a served checkpoint (`--early_exit_threshold`) |
| `run_nas.sh`, `modeling_ouro_nas.py`, `nas_compare.py` | 6-D router-architecture search driven by boptim-agent |

The optimisation loop itself (GP `gp_hedge`, `chatgpt`, `qwen`, `claude`, `random`) lives in
`boptim-agent/objective/ouro_*.py`; point it here with `--torchtitan_dir`.

**2. Terminal-Bench SFT.** Build a terminus-2 conversation set, fine-tune, export, serve, evaluate.

| Stage | Scripts |
|---|---|
| Data | `build_tb80_sft.py` (verified TB-2 leaderboard trajectories + domain-reweighted TerminalTraj), `build_tb80sft_v3.py` + `skill_mine.py` (thicken thin skills), `build_skill_sft.py` (one skill concentrated to ~40 %), `build_oracle12.py` (train-on-test diagnostic), `audit_skill_coverage.py`, `dist_compare.py` |
| Train | `run_v3.sh` (1.4B, 20k steps), `run_26b.sh` (2.6B), `run_skill.sh` / `run_skill_26b.sh` + `run_all_skills_26b.sh` (per-skill probes continuing from a checkpoint) |
| Serve + eval | `eval_parallel.sh` (12-task subset sharded over free GPUs), `eval_gated.sh` + `verify_agent.py` (TerminusVerify: refuses a `task_complete` that ran no commands), `eval80_26b.sh` (full 80 tasks, one server per GPU), `tally12.py`, `tally80.py`, `partial_credit.py` |
| Teacher generation | `convert_harbor.py` (Nemotron synthetic tasks to TB format), `gen_traj.sh` / `gen_all.sh` (a strong API model as the agent, rejection-sampled by each task's own pytest suite), `harvest_traj.py` (keep `is_resolved` trials as SFT rows) |

The benchmark harness (`terminal-bench 0.2.18`, local `tb_tasks/` copy of terminal-bench-core 0.1.1) is a
separate checkout at `~/terminal_bench_eval`; the drivers assume that layout.

**3. Qwen BO-agent distillation.** `qwen_bo_server.py` (transformers-only OpenAI-compatible server),
`rationalize.py`, `build_bo_sft.sh`, `run_bo_sft.sh` (config `qwen3_1_7b_bo_sft`), `bo_compare.py`; trajectory
generation is `gen_bo_traj.py` in boptim-agent.

### Results at a glance

| Experiment | Result |
|---|---|
| Exit gate on HumanEval, best tuned point (gp_hedge) | 1.229 loops / pass@1 0.7195: 3.25x faster than full depth, but the untuned gate at threshold 0.111 already gives 1.248 / 0.7256, so no tuned point beats the baseline |
| Terminal-Bench, Ouro-1.4B-Thinking | pretrained 0/80 and 0/24; every SFT recipe (1k to 20k steps, five data mixes) lands at 3-5/24 and 2-4/80 |
| Per-skill data matching (19 skills at 1.4B, 10 at 2.6B) | no task solved on both attempts; a handful of 1/2 flips at the noise floor |
| Ouro-2.6B-Thinking, same data and recipe | 6/24 ungated, 8/24 gated, 6/80; first checkpoint to solve `crack-7z-hash.easy` |
| Teacher generation (deepseek, pass@4, all-tests-pass filter) | 44 verified on-distribution trajectories after dedup; the 2.6B run on them is the next step |

Read: model capacity moved the benchmark, data curation did not. Details, settings and every per-run number are on the
[experiments page](https://claude.ai/artifact/RSGyFVwqfHdoFTnoBGNKib).

### Practical notes

- Single-GPU only for now: FSDP checkpoint load hangs for the 2.6B flavor, and seq-4096 runs OOM when a
  co-tenant holds ~15 GB of the card. `run_skill*.sh` pick the freest card and accept `SEQLEN=2048`.
- `WANDB_MODE=disabled` (or online) for long runs; offline mode captured stdout and froze the log once.
- A torchtitan checkpoint saves weights only. Export with `scripts/export_dcp_to_hf.py` and copy
  `config.json` / tokenizer / modeling files from the base assets into the serving dir, or the eval server will
  not load it.

### Slurm training (`ouro-simple` branch)

`run_ouro_train.slurm` wraps `run_train.sh` for a Slurm cluster. Edit its `#SBATCH` lines (account, partition,
GPUs, memory, time, log paths) and the `source .venv/bin/activate` line for your site, then:

```bash
mkdir -p logs
sbatch run_ouro_train.slurm
CONFIG=ouro_debugmodel NGPU=1 sbatch run_ouro_train.slurm --training.steps=100   # overrides pass through
```

Environment variables: `MODULE` (default `ouro`), `CONFIG` (default `ouro_1_4b`), `NGPU` (default `2`). WandB is on
by default in the Slurm script.


## Citation

We provide a detailed look into the parallelisms and optimizations available in `torchtitan`, along with summary advice on when to use various techniques.

[TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training](https://openreview.net/forum?id=SFN6Wm7YBI)
```
@inproceedings{
   liang2025torchtitan,
   title={TorchTitan: One-stop PyTorch native solution for production ready {LLM} pretraining},
   author={Wanchao Liang and Tianyu Liu and Less Wright and Will Constable and Andrew Gu and Chien-Chin Huang and Iris Zhang and Wei Feng and Howard Huang and Junjie Wang and Sanket Purandare and Gokul Nadathur and Stratos Idreos},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=SFN6Wm7YBI}
}
```


## License

Source code is made available under a [BSD 3 license](./LICENSE), however you may have other legal obligations that govern your use of other content linked in this repository, such as the license or terms of service for third-party data and models.
