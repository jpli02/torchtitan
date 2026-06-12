# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch.distributed as dist

from torchtitan.config import Configurable
from torchtitan.tools.logging import logger

# Path to scripts/evaluate_humaneval.py relative to the repo root
_DEFAULT_SCRIPT = str(
    Path(__file__).resolve().parent.parent.parent / "scripts" / "evaluate_humaneval.py"
)


class HumanEvalEvaluator(Configurable):
    """Runs HumanEval pass@1 evaluation after each checkpoint save.

    On multi-GPU runs only rank 0 spawns the subprocess; other ranks skip.
    When async_eval=True, evaluation runs in the background while training
    continues.  The next call to evaluate() (or trainer close()) waits for
    the previous subprocess to finish before starting a new one.

    Note: the evaluation subprocess needs access to a GPU.  When training
    uses all available GPUs set CUDA_VISIBLE_DEVICES in the environment to
    point the subprocess at a specific device, or use async_eval=False.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        """Enable HumanEval pass@1 evaluation after each checkpoint save."""

        module: str = ""
        """Trainer module name, e.g. 'ouro'.  Required when enable=True."""

        config: str = ""
        """Trainer config name, e.g. 'ouro_1_4b'.  Required when enable=True."""

        limit: int = 0
        """Number of HumanEval problems to evaluate (0 = all 164)."""

        max_gen_toks: int = 512
        """Maximum new tokens to generate per problem."""

        output_dir: str = "humaneval"
        """Sub-directory inside dump_folder to store per-step JSON results."""

        async_eval: bool = True
        """
        Run the evaluation subprocess in the background so training is not
        blocked.  The previous subprocess is waited on before a new one starts.
        """

        script_path: str = _DEFAULT_SCRIPT
        """
        Absolute path to scripts/evaluate_humaneval.py.
        Override if the script lives somewhere else.
        """

    def __init__(self, config: Config, *, dump_folder: str) -> None:
        if config.enable:
            if not config.module:
                raise ValueError("humaneval_eval.module must be set when enable=True.")
            if not config.config:
                raise ValueError("humaneval_eval.config must be set when enable=True.")
        self.config = config
        self.dump_folder = dump_folder
        self._proc: subprocess.Popen | None = None

    def evaluate(self, checkpoint_id: str, step: int) -> None:
        """Launch HumanEval evaluation for the checkpoint saved at *step*.

        Only executes on rank 0.  Waits for any previous async subprocess
        before spawning a new one.
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        self.wait()

        output_path = os.path.join(
            self.dump_folder,
            self.config.output_dir,
            f"step-{step}.json",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cmd = [
            sys.executable,
            self.config.script_path,
            "--module", self.config.module,
            "--config", self.config.config,
            "--checkpoint", checkpoint_id,
            "--max_gen_toks", str(self.config.max_gen_toks),
            "--output_path", output_path,
        ]
        if self.config.limit > 0:
            cmd += ["--limit", str(self.config.limit)]

        logger.info(
            "HumanEval: launching evaluation for step %d (checkpoint: %s)",
            step,
            checkpoint_id,
        )

        if self.config.async_eval:
            self._proc = subprocess.Popen(cmd)
        else:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.warning(
                    "HumanEval evaluation for step %d failed (returncode=%d).",
                    step,
                    result.returncode,
                )
            else:
                logger.info("HumanEval evaluation for step %d completed.", step)

    def wait(self) -> None:
        """Block until the background evaluation subprocess finishes, if any."""
        if self._proc is not None:
            self._proc.wait()
            if self._proc.returncode != 0:
                logger.warning(
                    "Background HumanEval evaluation exited with code %d.",
                    self._proc.returncode,
                )
            else:
                logger.info("Background HumanEval evaluation finished successfully.")
            self._proc = None
