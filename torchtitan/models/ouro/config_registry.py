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
            export_dtype="float16",
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
