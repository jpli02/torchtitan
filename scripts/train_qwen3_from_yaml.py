#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Train Qwen3 with a custom architecture from a YAML config file.

Use this when an LLM or tool generates a YAML config. The schema is defined in
torchtitan/models/qwen3/LLM_CONFIG_INTERFACE.md.

Example:
    torchrun --nproc_per_node=1 torchtitan/scripts/train_qwen3_from_yaml.py \\
        --config-yaml torchtitan/models/qwen3/configs/example.yaml

    torchrun --nproc_per_node=4 torchtitan/scripts/train_qwen3_from_yaml.py \\
        --config-yaml my_custom_config.yaml
"""

import argparse
import os
import sys

import torch

# Add torchtitan root to path when running as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TORCHTITAN_ROOT = os.path.dirname(_SCRIPT_DIR)
if _TORCHTITAN_ROOT not in sys.path:
    sys.path.insert(0, _TORCHTITAN_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Qwen3 from YAML config (see LLM_CONFIG_INTERFACE.md)"
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    from torchtitan.tools.logging import init_logger, logger

    init_logger()

    from torchtitan.models.qwen3.yaml_config import config_from_yaml

    config = config_from_yaml(args.config_yaml)
    trainer = None

    try:
        if config.comm.mode == "local_tensor":
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        trainer = config.build()

        if config.checkpoint.create_seed_checkpoint:
            assert int(os.environ.get("WORLD_SIZE", "1")) == 1, (
                "Must create seed checkpoint using a single device."
            )
            assert config.checkpoint.enable, (
                "Must enable checkpointing when creating a seed checkpoint."
            )
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
