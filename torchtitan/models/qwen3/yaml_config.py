# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load Qwen3 training config from YAML for LLM-generated configurations."""

from pathlib import Path

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import build_custom_qwen3_spec

# Default values matching qwen3_custom_gsm8k
DEFAULTS = {
    "model": {
        "dim": 1024,
        "n_layers": 28,
        "n_heads": 16,
        "n_kv_heads": 8,
        "head_dim": 128,
        "ffn_hidden_dim": None,  # 3*dim when None
        "vocab_size": 151936,
        "max_seq_len": 4096,
        "norm_eps": 1e-6,
        "enable_weight_tying": True,
    },
    "training": {
        "local_batch_size": 4,
        "seq_len": 2048,
        "steps": 1000,
    },
    "dataset": "gsm8k",
    "optimizer": {"lr": 3e-4, "weight_decay": 0.1},
    "lr_scheduler": {
        "warmup_steps": 100,
        "decay_ratio": 0.1,
        "decay_type": "cosine",
        "min_lr_factor": 0.1,
    },
    "hf_assets_path": "./assets/hf/Qwen3-0.6B",
    "dump_folder": "./outputs",
    "checkpoint": {"interval": 100, "export_dtype": "float16"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def config_from_yaml(path: str | Path) -> Trainer.Config:
    """Build Trainer.Config from a YAML file.

    The YAML must follow the schema in LLM_CONFIG_INTERFACE.md.
    Omitted fields use defaults.

    Args:
        path: Path to YAML config file.

    Returns:
        Trainer.Config ready for training.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML config. Install with: pip install pyyaml")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data:
        data = {}

    # Merge with defaults (shallow merge for top-level, deep for nested)
    merged = _deep_merge(DEFAULTS, data)

    model_cfg = merged["model"]
    training_cfg = merged["training"]
    opt_cfg = merged["optimizer"]
    lr_cfg = merged["lr_scheduler"]
    ckpt_cfg = merged["checkpoint"]

    # Resolve ffn_hidden_dim
    ffn_hidden_dim = model_cfg.get("ffn_hidden_dim")
    if ffn_hidden_dim is None:
        ffn_hidden_dim = 3 * model_cfg["dim"]

    model_spec = build_custom_qwen3_spec(
        dim=model_cfg["dim"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        n_kv_heads=model_cfg["n_kv_heads"],
        head_dim=model_cfg["head_dim"],
        ffn_hidden_dim=ffn_hidden_dim,
        vocab_size=model_cfg["vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        norm_eps=model_cfg["norm_eps"],
        enable_weight_tying=model_cfg["enable_weight_tying"],
    )

    return Trainer.Config(
        hf_assets_path=merged["hf_assets_path"],
        dump_folder=merged["dump_folder"],
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset=merged["dataset"],
            infinite=True,
        ),
        optimizer=OptimizersContainer.Config(
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.1),
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=lr_cfg["warmup_steps"],
            decay_ratio=lr_cfg["decay_ratio"],
            decay_type=lr_cfg["decay_type"],
            min_lr_factor=lr_cfg["min_lr_factor"],
        ),
        training=TrainingConfig(
            local_batch_size=training_cfg["local_batch_size"],
            seq_len=training_cfg["seq_len"],
            steps=training_cfg["steps"],
        ),
        checkpoint=CheckpointManager.Config(
            interval=ckpt_cfg["interval"],
            last_save_model_only=False,
            export_dtype=ckpt_cfg["export_dtype"],
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )
