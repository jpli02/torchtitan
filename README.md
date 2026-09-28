# Ouro on torchtitan

Training, exit-gate tuning and Terminal-Bench agent fine-tuning for
[ByteDance Ouro](https://huggingface.co/ByteDance/Ouro-1.4B), a looped ("universal") Transformer that
re-applies its layers up to 4 times per token and learns an early-exit gate. The trainer underneath is
[torchtitan](https://github.com/pytorch/torchtitan); this repository adds the Ouro model, its configs, a
2.6B flavor, an 8-bit optimizer path, and the scripts behind three lines of experiments:

- **Exit-gate search**: Bayesian optimisation of the gate's training margin and exit threshold on HumanEval
  and MBPP, plus a router-architecture NAS, driven from [`boptim-agent`](https://github.com/jpli02/boptim-agent).
- **Terminal-Bench SFT**: fine-tuning Ouro-1.4B-Thinking and Ouro-2.6B-Thinking as a terminus-2 terminal
  agent, with data builders, per-skill probes, a gated evaluation harness and teacher trajectory generation.
- **Qwen BO-agent distillation**: turning a GP optimiser's trajectories into an SFT set for Qwen3-1.7B.

Every number, setting and conclusion is on the
[boptim Experiments page](https://claude.ai/artifact/RSGyFVwqfHdoFTnoBGNKib). The research scripts are
indexed in [`research/README.md`](research/README.md).

## Where things live

| Branch | Contents |
|---|---|
| `main` | Model, configs, and every research script under [`research/`](research/README.md): `bo_gate/` (exit-gate BO and HumanEval/MBPP evals), `terminal_sft/{data,train,eval,gen}/`, `qwen_bo_agent/`, `pretrain/`, `docs/`. |
| `worktree-ouro-terminal-sft` | The working branch the experiments were run from; merged into `main`. |
| `ouro-boptim` | The checkout used by [`boptim-agent`](https://github.com/jpli02/boptim-agent) for the exit-gate Bayesian-optimisation objective. |
| `ouro-simple` | The original Slurm training wrapper (`research/pretrain/run_ouro_train.slurm`); see the end of this section. |

Model code: [`torchtitan/models/ouro/`](torchtitan/models/ouro) (`model.py`, `parallelize.py`, `state_dict_adapter.py`,
`config_registry.py`). HF assets are expected under `./assets/hf/<model>` (`Ouro-1.4B`, `Ouro-1.4B-Thinking`, ...);
`scripts/download_hf_assets.py` fetches them.

## Model flavors and configs

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

## Quick start (single node, no Slurm)

```bash
pip install -r requirements.txt            # plus torchao for AdamW8bit
# smoke test
MODULE=ouro CONFIG=ouro_debugmodel NGPU=1 ./run_train.sh --training.steps=20
# terminal-agent SFT of Ouro-1.4B-Thinking on a local JSONL of terminus-2 conversations
OURO_SFT_LOCAL_JSONL=/path/to/train.jsonl MODULE=ouro CONFIG=ouro_1_4b_thinking_terminal_sft NGPU=1 ./run_train.sh --training.steps=20000
```

The research drivers below call `torchtitan.train` directly (not `run_train.sh`) so that the wrapper's default
batch-size arguments do not override the config; read `research/terminal_sft/train/run_v3.sh` for the canonical invocation.

## Pipelines

**1. Exit-gate Bayesian optimisation (HumanEval / MBPP).** Each BO evaluation runs `ouro_1_4b_sft` for 100
steps at a proposed `adaptive_gamma`, exports the gate, and scores loops-per-token and pass@1.
The driver is the `ask -> evaluate -> tell` loop in `boptim-agent/main.py`; `evaluate` is what calls into this
repository:

```python
for i in range(start_iter, max_iter):
    points, ask_explanation = optimizer.ask(n_points=batch_size)   # GP acquisition or an LLM's JSON proposal
    if len(points) == 0:
        break                                                      # optimizer declared itself done
    fvals = objective_fn.evaluate_batch(points)                    # Ouro: 100-step gate SFT + benchmark eval
    result, tell_explanation = optimizer.tell(points, [sign * fv for fv in fvals])   # sign=-1 when maximizing
```


| Script | Role |
|---|---|
| `scripts/evaluate_humaneval_evalplus.py`, `scripts/evaluate_mbpp.py` | benchmark evals with early-exit threshold, KV-cache and prompt-format flags |
| `scripts/export_dcp_to_hf.py`, `scripts/extract_router.py` | DCP checkpoint to HF serving dir; pull the gate out as a `.safetensors` |
| `scripts/ouro_openai_server.py` | OpenAI-compatible `/v1/chat/completions` server for a served checkpoint (`--early_exit_threshold`) |
| `research/bo_gate/run_nas.sh`, `research/bo_gate/modeling_ouro_nas.py`, `research/bo_gate/nas_compare.py` | 6-D router-architecture search driven by boptim-agent |

The optimisation loop itself (GP `gp_hedge`, `chatgpt`, `qwen`, `claude`, `random`) lives in
[`boptim-agent`](https://github.com/research4pan/boptim-agent), which vendors this repository as its
`torchtitan` submodule; its `objective/ouro_*.py` call back into this checkout via `--torchtitan_dir`.
Scripts here find boptim-agent through `BOPTIM_REPO`, defaulting to the parent directory of this checkout.

### Running the BO loop

The loop runs from boptim-agent (the parent checkout) and calls back into this checkout for every evaluation.

```bash
# 1. one-time setup, from the boptim-agent root (this repo is its torchtitan/ submodule):
#    boptim-agent's venv (scikit-optimize, openai, ...) and torchtitan's venv with Ouro assets
#    under torchtitan/assets/hf/Ouro-1.4B
cd ..            # from torchtitan/ up to the boptim-agent root
uv sync
set -a; . ~/.boptim_keys.env; set +a          # OPENAI_API_KEY / QWEN_* only for the LLM optimisers

# 2. smoke test of the whole train -> checkpoint -> HumanEval -> score chain (20 SFT steps, 20 problems)
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline ./.venv/bin/python run_experiment.py \
    --config configs/ouro_humaneval_gp_smoke_local.yaml

# 3. the real search: 10 BO iterations, each = 100-step gate SFT on GPU 0 + full-164 HumanEval
#    sharded over GPUs 0,1 (~40 min per point, ~7 h total)
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline ./.venv/bin/python run_experiment.py \
    --config configs/ouro_humaneval_gp_full_local.yaml

# 4. same search with an LLM optimiser, resumable across crashes
CUDA_VISIBLE_DEVICES=0,1 ./.venv/bin/python main.py --optim chatgpt --model gpt-5-mini --no-explanation \
    --objective ouro --torchtitan_dir torchtitan --resume \
    --ouro_train_config ouro_1_4b_sft --ouro_eval_config ouro_1_4b --ouro_steps 100 \
    --ouro_ngpu 1 --ouro_eval_ngpu 2 --ouro_search_threshold --ouro_baseline_pass1 0.68 \
    --bounds 0 0.0 1.0 1 0.5 1.0 --max_iter 10 --optimizer-seed 42

# 5. router-architecture NAS (6-D space, gp_hedge / claude / chatgpt / qwen)
OPTIM=gp_hedge ITERS=25 GPU=0 TAG=nas_gp bash torchtitan/research/bo_gate/run_nas.sh
```

What the config keys mean: `torchtitan_dir` is this checkout (the YAMLs say `torchtitan`, resolved against the
boptim-agent root; `TORCHTITAN_DIR` or `--torchtitan_dir` override it); `ouro_train_config` / `ouro_eval_config` are the configs above;
`ouro_steps` is the gate SFT length; `ouro_ngpu` / `ouro_eval_ngpu` map shard *i* to the *i*-th visible GPU;
`bounds` is `[lo hi]` per dimension for `adaptive_gamma` and `early_exit_threshold`; `ouro_baseline_pass1`
is the accuracy floor below which a point scores `ouro_loop_penalty`. Each evaluation leaves a run dir under
`torchtitan/outputs/boptim/`, the BO state under `runs/state_*.json` (what `--resume` replays), and the
log under `logs/`. Two cautions from the experiments: set the floor from the
untuned gate's threshold sweep (0.7256 on HumanEval, not the full-depth 0.68), and the MBPP objective still
ignores the searched threshold (see the experiments page).

**2. Terminal-Bench SFT.** Build a terminus-2 conversation set, fine-tune, export, serve, evaluate.

| Stage | Scripts |
|---|---|
| Data | `research/terminal_sft/data/build_tb80_sft.py` (verified TB-2 leaderboard trajectories + domain-reweighted TerminalTraj), `research/terminal_sft/data/build_tb80sft_v3.py` + `research/terminal_sft/data/skill_mine.py` (thicken thin skills), `research/terminal_sft/data/build_skill_sft.py` (one skill concentrated to ~40 %), `research/terminal_sft/data/build_oracle12.py` (train-on-test diagnostic), `research/terminal_sft/data/audit_skill_coverage.py`, `research/terminal_sft/data/dist_compare.py` |
| Train | `research/terminal_sft/train/run_v3.sh` (1.4B, 20k steps), `research/terminal_sft/train/run_26b.sh` (2.6B), `research/terminal_sft/train/run_skill.sh` / `research/terminal_sft/train/run_skill_26b.sh` + `research/terminal_sft/train/run_all_skills_26b.sh` (per-skill probes continuing from a checkpoint) |
| Serve + eval | `research/terminal_sft/eval/eval_parallel.sh` (12-task subset sharded over free GPUs), `research/terminal_sft/eval/eval_gated.sh` + `research/terminal_sft/eval/verify_agent.py` (TerminusVerify: refuses a `task_complete` that ran no commands), `research/terminal_sft/eval/eval80_26b.sh` (full 80 tasks, one server per GPU), `research/terminal_sft/eval/tally12.py`, `research/terminal_sft/eval/tally80.py`, `research/terminal_sft/eval/partial_credit.py` |
| Teacher generation | `research/terminal_sft/gen/convert_harbor.py` (Nemotron synthetic tasks to TB format), `research/terminal_sft/gen/gen_traj.sh` / `research/terminal_sft/gen/gen_all.sh` (a strong API model as the agent, rejection-sampled by each task's own pytest suite), `research/terminal_sft/gen/harvest_traj.py` (keep `is_resolved` trials as SFT rows) |

The benchmark harness (`terminal-bench 0.2.18`, local `tb_tasks/` copy of terminal-bench-core 0.1.1) is a
separate checkout at `~/terminal_bench_eval`; the drivers assume that layout.

### Running a Terminal-Bench experiment

One driver does the whole chain: train, export a serving dir, evaluate. Example, the tb80sft v3 recipe
for Ouro-1.4B-Thinking: 20k SFT steps on one GPU (lr 2e-5, seq 4096, about 3.3 s/step, roughly 18 h),
then the 12-task subset evaluated twice per task with one Ouro server per free GPU.

```bash
# from the torchtitan checkout; needs ~/terminal_bench_eval (terminal-bench 0.2.18 + tb_tasks/) and
# the tb80sft v3 JSONL (research/terminal_sft/data/build_tb80sft_v3.py)
bash research/terminal_sft/train/run_v3.sh                              # STEPS=... to shorten
```

Results land in `~/.claude/jobs/c0d2da0a/tmp/tb80sft_v3_chain.txt` (training tail, then per-task
resolved / unresolved) with per-trial JSON under `/tmp/tb_runs/<run>/`. To evaluate an existing checkpoint
only, or to use the gated agent, call the eval stage directly:

```bash
CKPT=/path/to/serving_dir BASENAME=my_ckpt GATE=1 bash research/terminal_sft/eval/eval_parallel.sh
```

`run_26b.sh` is the same chain for Ouro-2.6B-Thinking, `run_skill.sh` / `run_skill_26b.sh` the per-skill
probes, and `eval80_26b.sh` the full 80-task protocol.

**3. Qwen BO-agent distillation.** `research/qwen_bo_agent/qwen_bo_server.py` (transformers-only OpenAI-compatible server),
`research/qwen_bo_agent/rationalize.py`, `research/qwen_bo_agent/build_bo_sft.sh`, `research/qwen_bo_agent/run_bo_sft.sh` (config `qwen3_1_7b_bo_sft`), `research/qwen_bo_agent/bo_compare.py`; trajectory
generation is `gen_bo_traj.py` in boptim-agent (a copy lives in `research/qwen_bo_agent/`).

## Results at a glance

| Experiment | Result |
|---|---|
| Exit gate on HumanEval, best tuned point (gp_hedge) | 1.229 loops / pass@1 0.7195: 3.25x faster than full depth, but the untuned gate at threshold 0.111 already gives 1.248 / 0.7256, so no tuned point beats the baseline |
| Terminal-Bench, Ouro-1.4B-Thinking | pretrained 0/80 and 0/24; every SFT recipe (1k to 20k steps, five data mixes) lands at 3-5/24 and 2-4/80 |
| Per-skill data matching (19 skills at 1.4B, 10 at 2.6B) | no task solved on both attempts; a handful of 1/2 flips at the noise floor |
| Ouro-2.6B-Thinking, same data and recipe | 6/24 ungated, 8/24 gated, 6/80; first checkpoint to solve `crack-7z-hash.easy` |
| Teacher generation (deepseek, pass@4, all-tests-pass filter) | 44 verified on-distribution trajectories after dedup; the 2.6B run on them is the next step |

Read: model capacity moved the benchmark, data curation did not. Details, settings and every per-run number are on the
[experiments page](https://claude.ai/artifact/RSGyFVwqfHdoFTnoBGNKib).

## Practical notes

- Single-GPU only for now: FSDP checkpoint load hangs for the 2.6B flavor, and seq-4096 runs OOM when a
  co-tenant holds ~15 GB of the card. `run_skill*.sh` pick the freest card and accept `SEQLEN=2048`.
- `WANDB_MODE=disabled` (or online) for long runs; offline mode captured stdout and froze the log once.
- A torchtitan checkpoint saves weights only. Export with `scripts/export_dcp_to_hf.py` and copy
  `config.json` / tokenizer / modeling files from the base assets into the serving dir, or the eval server will
  not load it.

## Slurm training (`ouro-simple` branch)

`research/pretrain/run_ouro_train.slurm` wraps `run_train.sh` for a Slurm cluster. Edit its `#SBATCH` lines (account, partition,
GPUs, memory, time, log paths) and the `source .venv/bin/activate` line for your site, then:

```bash
mkdir -p logs
sbatch research/pretrain/run_ouro_train.slurm
CONFIG=ouro_debugmodel NGPU=1 sbatch research/pretrain/run_ouro_train.slurm --training.steps=100   # overrides pass through
```

Environment variables: `MODULE` (default `ouro`), `CONFIG` (default `ouro_1_4b`), `NGPU` (default `2`). WandB is on
by default in the Slurm script.

## Installation

```bash
# this repo is the `torchtitan` submodule of boptim-agent; clone the parent to get both
git clone --recurse-submodules git@github.com:research4pan/boptim-agent.git && cd boptim-agent/torchtitan
pip install -r requirements.txt      # a recent PyTorch nightly, plus torchao for AdamW8bit
```

## License

Source code is made available under a [BSD 3 license](./LICENSE); model weights and datasets referenced here
carry their own terms.
