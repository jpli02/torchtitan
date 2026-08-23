# Ouro model training configs

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry


def ouro_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            folder="/projects/bdjz/jli37/checkpoints",
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )

def ouro_1_4b() -> Trainer.Config:
    """ByteDance/Ouro-1.4B from HuggingFace."""
    return Trainer.Config(
        hf_assets_path="./assets/hf/Ouro-1.4B",  # ByteDance/Ouro-1.4B
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("1.4B"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="swe_rebench_openhands"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            # steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="/projects/bdjz/jli37/checkpoints",
            interval=100,
            last_save_model_only=False,
            # The released Ouro checkpoint is bf16. Keep HF exports in bf16 too
            # so KV-cache evals do not compare fp16-exported weights against the
            # TorchTitan/DCP path.
            export_dtype="bfloat16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )


def ouro_1_4b_sft() -> Trainer.Config:
    """SFT variant of ouro_1_4b: loss is masked to assistant turns only, Stage II adaptive gate."""
    import dataclasses

    cfg = ouro_1_4b()
    cfg.dataloader = HuggingFaceTextDataLoader.Config(
        dataset="open_code_reasoning_sft"
    )
    cfg.optimizer = OptimizersContainer.Config(lr=3e-4)
    # The Stage-II adaptive-gate loss hyperparameters (gamma = improvement
    # margin, k = sigmoid slope) are tunable from the environment so an outer
    # search (the boptim-agent BO loop) can sweep them without a CLI override
    # path -- tyro does not expose model_spec fields. Defaults match
    # OuroModel.Config (gamma=0.005, k=50.0).
    import os
    model_overrides = {"ouro_loss_stage": "stage2_adaptive"}
    if (g := os.environ.get("OURO_ADAPTIVE_GAMMA")) is not None:
        model_overrides["adaptive_gamma"] = float(g)
    if (k := os.environ.get("OURO_ADAPTIVE_K")) is not None:
        model_overrides["adaptive_k"] = float(k)
    cfg.model_spec = dataclasses.replace(
        cfg.model_spec,
        model=dataclasses.replace(cfg.model_spec.model, **model_overrides),
    )
    # SFT only trains the adaptive exit gate (backbone frozen), so the backbone
    # must start from the pretrained ByteDance/Ouro-1.4B weights rather than
    # random init. Without this, the gate is tuned over a random backbone and
    # the model emits garbage at eval (HumanEval pass@1 = 0).
    cfg.checkpoint = dataclasses.replace(
        cfg.checkpoint,
        initial_load_path=cfg.hf_assets_path,  # ./assets/hf/Ouro-1.4B
        initial_load_in_hf=True,
        initial_load_model_only=True,
    )
    return cfg


def ouro_1_4b_thinking_terminal_sft() -> Trainer.Config:
    """Full-backbone SFT of Ouro-1.4B-Thinking on the terminal-agent mixture.

    Distinct from ouro_1_4b_sft in two ways that matter:

    1. ``ouro_loss_stage`` is left at OuroModel.Config's default,
       ``stage1_entropy`` -- Ouro's own Stage-I objective
       (L = sum_t p(t|x) L^(t) - beta*H(p)), which trains every recurrence depth
       weighted by the exit distribution and honours the SFT label mask.
       Critically it does NOT freeze anything. ouro_1_4b_sft instead selects
       ``stage2_adaptive``, which freezes every parameter except the exit gate
       (model.py: ``if ouro_loss_stage == "stage2_adaptive": ... requires_grad_(False)``)
       -- 2049 trainable params out of 1.4B. That is the right setup for tuning
       *when* the model exits its recurrence loop, and completely wrong for
       teaching it *what* to emit. Terminal-Bench failures are behavioural (prose
       or malformed JSON the harness cannot parse into actions), so the backbone
       has to move. Stage I also emits per-recurrence-step CE as aux metrics,
       which is exactly the signal worth watching on W&B for this run.
    2. It starts from the Thinking checkpoint, not the base one, since that is
       the model actually being evaluated.

    LR is 2e-5, not the 3e-4 the gate-only configs use: 3e-4 is a reasonable rate
    for a 2k-parameter head trained from scratch, but roughly an order of
    magnitude above the usual full-finetune range for a 1.4B model and would
    scorch the pretrained weights in a 1k-step run.
    """
    import dataclasses

    cfg = ouro_1_4b()
    cfg.hf_assets_path = "./assets/hf/Ouro-1.4B-Thinking"
    cfg.dataloader = HuggingFaceTextDataLoader.Config(dataset="terminal_agent_sft")
    cfg.optimizer = OptimizersContainer.Config(lr=2e-5)
    # Warm up over the first 5% of the run rather than the 2 steps the other
    # ouro configs use: with the full backbone unfrozen, the first optimizer
    # steps at full LR are where a finetune most easily damages pretrained
    # weights. Cosine decay to a small floor for a clean 1k-step schedule.
    cfg.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=50,
        decay_ratio=0.9,
        decay_type="cosine",
        min_lr_factor=0.1,
    )
    cfg.training = TrainingConfig(
        local_batch_size=1,
        # 4096, not the 8192 agent trajectories would ideally want, because the
        # Stage-I loss is what sets the memory ceiling here: it materialises
        # stacked_step_logits [B, S, V, T] (V=49152, T=4 recurrence steps) and
        # then casts each step's slice to fp32 inside the CE loop, all of which
        # autograd retains for backward. At 8192 that is ~3.2GB stacked plus
        # 4 x 1.6GB fp32 copies on top of ~22GB of fp32 params/grads/Adam
        # states, and it OOMs a 46GB A6000 (measured, not estimated). Halving
        # the sequence halves every one of those terms. Sequence packing means
        # no trajectory data is dropped by this -- long rows simply span more
        # than one training sequence.
        seq_len=4096,
        steps=1000,
    )
    # Full (not selective) activation checkpointing: Ouro runs its decoder stack
    # total_ut_steps=4 times per token, so activation memory is ~4x a same-size
    # non-recurrent model and is the other half of the budget the loss above
    # competes with. Recompute is the cheaper trade here.
    cfg.activation_checkpoint = ActivationCheckpointConfig(mode="full")
    cfg.metrics = MetricsProcessor.Config(log_freq=10, enable_wandb=True)
    cfg.checkpoint = dataclasses.replace(
        cfg.checkpoint,
        initial_load_path=cfg.hf_assets_path,
        initial_load_in_hf=True,
        initial_load_model_only=True,
        interval=250,
        # Export the final checkpoint straight to HF safetensors so it can be
        # served by scripts/ouro_openai_server.py and pushed to the Hub without
        # a separate DCP->HF conversion pass.
        last_save_model_only=True,
        last_save_in_hf=True,
        export_dtype="bfloat16",
    )
    return cfg
