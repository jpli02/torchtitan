# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Any, TypeAlias

import torch
import torch.nn.functional as F

from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss with sum reduction for token-based normalization."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def build_cross_entropy_loss(compile_config: CompileConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = cross_entropy_loss
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common MSE loss function with sum reduction for Transformer models training."""
    return torch.nn.functional.mse_loss(
        pred.float(), labels.float().detach(), reduction="sum"
    )


def build_mse_loss(compile_config: CompileConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = mse_loss
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn


def loop_lm_entropy_regularized_loss_sum(
    pred: dict[str, Any],
    labels: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """
    Stage I: L = sum_t p(t|x) L^(t) - beta * H(p), minimized (sum over valid tokens).

    pred must contain ``stacked_exit_pdf`` [B, S, T] and ``stacked_step_logits``
    [B, S, V, T]. Returns a scalar sum for the same normalization contract as
    ``cross_entropy_loss`` (divide by global valid tokens in the trainer).
    """
    stacked_exit_pdf = pred["stacked_exit_pdf"]
    step_logits = pred["stacked_step_logits"]
    b, s, v, t_max = step_logits.shape
    valid = (labels != IGNORE_INDEX).to(step_logits.dtype)

    ce_steps = []
    for t in range(t_max):
        ce_t = F.cross_entropy(
            step_logits[:, :, :, t].reshape(-1, v).float(),
            labels.reshape(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).view(b, s)
        ce_steps.append(ce_t)
    ce_per_step = torch.stack(ce_steps, dim=-1)
    p = stacked_exit_pdf.to(dtype=torch.float32)
    weighted_ce = (p * ce_per_step).sum(dim=-1)
    task_sum = (valid * weighted_ce).sum()

    # H(p) = -sum p log p = sum_i entr(p_i); entr(0)=0
    H = torch.special.entr(p.clamp_min(0.0)).sum(dim=-1)
    entropy_sum = (valid * H).sum()
    return task_sum - beta * entropy_sum


def loop_lm_adaptive_gate_loss_sum(
    pred: dict[str, Any],
    labels: torch.Tensor,
    *,
    k: float = 50.0,
    gamma: float = 0.005,
) -> torch.Tensor:
    """
    Stage II: binary cross-entropy between ideal continuation label w^(t) and
    predicted continuation (1 - lambda^(t)), averaged over sequence then over
    t = 2..Tmax with factor 1/Tmax. Per-step losses L^(t)_stop are detached from
    the LM so only the exit gate receives LM-driven gradients through lambda.
    """
    step_logits = pred["stacked_step_logits"].detach()
    gate_lambda = pred["gate_lambda"]
    t_max = int(pred.get("total_ut_steps", gate_lambda.shape[-1]))
    b, s, v, _t = step_logits.shape
    valid = (labels != IGNORE_INDEX).to(step_logits.dtype)

    l_stop = []
    for t in range(_t):
        l_t = F.cross_entropy(
            step_logits[:, :, :, t].reshape(-1, v).float(),
            labels.reshape(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).view(b, s)
        l_stop.append(l_t)
    L_stop = torch.stack(l_stop, dim=-1)

    eps = 1e-7
    total = gate_lambda.new_zeros(())
    for t_idx in range(1, _t):
        improvement = F.relu(L_stop[:, :, t_idx - 1] - L_stop[:, :, t_idx])
        w = torch.sigmoid(k * (improvement - gamma))
        lam = gate_lambda[:, :, t_idx].clamp(eps, 1.0 - eps)
        one_m = (1.0 - lam).clamp(eps, 1.0 - eps)
        bce = -(w * one_m.log() + (1.0 - w) * lam.log())
        total = total + (valid * bce).sum()

    return total / float(t_max)


def build_ouro_loss(compile_config: CompileConfig, **kwargs) -> LossFunction:
    """Ouro / LoopLM losses: Stage I (entropy-regularized) or Stage II (adaptive gate)."""
    model_config = kwargs.pop("model_config", None)
    del kwargs  # parallel_dims, ft_manager, etc.

    stage = "stage1_entropy"
    entropy_beta = 0.01
    adaptive_k = 50.0
    adaptive_gamma = 0.005
    if model_config is not None:
        stage = getattr(model_config, "ouro_loss_stage", stage)
        entropy_beta = float(getattr(model_config, "entropy_beta", entropy_beta))
        adaptive_k = float(getattr(model_config, "adaptive_k", adaptive_k))
        adaptive_gamma = float(getattr(model_config, "adaptive_gamma", adaptive_gamma))

    def ouro_loss_fn(pred: torch.Tensor | dict[str, Any], labels: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, dict):
            if stage == "stage1_entropy":
                return loop_lm_entropy_regularized_loss_sum(
                    pred, labels, beta=entropy_beta
                )
            if stage == "stage2_adaptive":
                return loop_lm_adaptive_gate_loss_sum(
                    pred, labels, k=adaptive_k, gamma=adaptive_gamma
                )
            return cross_entropy_loss(pred["logits"], labels)
        return cross_entropy_loss(pred, labels)

    loss_fn = ouro_loss_fn
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the Ouro loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn
