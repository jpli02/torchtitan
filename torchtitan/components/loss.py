# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(
    pred: torch.Tensor, labels: torch.Tensor, *, model=None, **kwargs
) -> torch.Tensor:
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


def mse_loss(
    pred: torch.Tensor, labels: torch.Tensor, *, model=None, **kwargs
) -> torch.Tensor:
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


# --- Loop LM (Ouro): exit distribution, Stage I / II ---------------------------------


def token_cross_entropy_per_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    logits: [B, S, V]
    labels: [B, S]
    return: per-token CE loss, [B, S]
    """
    b, s, v = logits.shape
    loss = F.cross_entropy(
        logits.reshape(b * s, v),
        labels.reshape(b * s),
        reduction="none",
        ignore_index=ignore_index,
    )
    return loss.reshape(b, s)


def reduce_masked_mean(
    x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    x:    [B, S] or compatible
    mask: [B, S], 1 for valid, 0 for ignored
    """
    x = x * mask
    denom = mask.sum().clamp_min(eps)
    return x.sum() / denom


def build_exit_distribution(exit_logits_list: list[torch.Tensor]):
    """
    Build p_phi(t|x) from instantaneous exit probabilities lambda_t.

    exit_logits_list: list of T tensors, each [B, S] or [B, S, 1]

    Returns:
        lambdas: [T, B, S]
        p_exit:  [T, B, S]
        survival: stacked S_0..S_{T-1}, shape [T, B, S]
    """
    lambdas = []
    for x in exit_logits_list:
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)
        lambdas.append(torch.sigmoid(x))
    lambdas = torch.stack(lambdas, dim=0)  # [T, B, S]

    t_max, b, s = lambdas.shape
    device = lambdas.device
    dtype = lambdas.dtype

    survival_list = [torch.ones(b, s, device=device, dtype=dtype)]  # S_0 = 1
    p_exit_list = []

    for t in range(t_max):
        if t < t_max - 1:
            p_t = lambdas[t] * survival_list[t]
            p_exit_list.append(p_t)
            survival_list.append(survival_list[t] * (1.0 - lambdas[t]))
        else:
            p_t = survival_list[t]
            p_exit_list.append(p_t)

    p_exit = torch.stack(p_exit_list, dim=0)  # [T, B, S]
    survival = torch.stack(survival_list, dim=0)  # [T, B, S]
    return lambdas, p_exit, survival


class LoopLMLoss(nn.Module):
    def __init__(
        self,
        beta_entropy: float = 0.05,
        gamma_improve: float = 0.005,
        k_improve: float = 50.0,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.beta_entropy = beta_entropy
        self.gamma_improve = gamma_improve
        self.k_improve = k_improve
        self.ignore_index = ignore_index

    def _valid_mask(self, labels: torch.Tensor) -> torch.Tensor:
        return (labels != self.ignore_index).float()

    def stage1_loss(
        self,
        step_logits_list: list[torch.Tensor],
        step_exit_logits_list: list[torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        t_steps = len(step_logits_list)
        assert t_steps == len(step_exit_logits_list), "Mismatch in recurrent steps."

        valid_mask = self._valid_mask(labels)  # [B, S]

        per_step_ce = []
        for logits in step_logits_list:
            ce = token_cross_entropy_per_token(
                logits, labels, ignore_index=self.ignore_index
            )
            per_step_ce.append(ce)
        per_step_ce = torch.stack(per_step_ce, dim=0)  # [T, B, S]

        lambdas, p_exit, _ = build_exit_distribution(step_exit_logits_list)

        expected_ce_per_token = (p_exit * per_step_ce).sum(dim=0)  # [B, S]
        eps = 1e-8
        entropy_per_token = -(p_exit * torch.log(p_exit.clamp_min(eps))).sum(dim=0)

        per_token_obj = (
            expected_ce_per_token - self.beta_entropy * entropy_per_token
        ) * valid_mask
        loss_sum = per_token_obj.sum()

        expected_ce = reduce_masked_mean(expected_ce_per_token, valid_mask)
        entropy = reduce_masked_mean(entropy_per_token, valid_mask)
        total_mean = expected_ce - self.beta_entropy * entropy

        avg_exit_step = reduce_masked_mean(
            sum((t + 1) * p_exit[t] for t in range(t_steps)),
            valid_mask,
        )

        return {
            "loss_sum": loss_sum,
            "loss": total_mean,
            "expected_ce": expected_ce,
            "entropy": entropy,
            "avg_exit_step": avg_exit_step,
            "p_exit": p_exit.detach(),
            "per_step_ce": per_step_ce.detach(),
            "lambdas": lambdas.detach(),
        }

    def stage2_loss(
        self,
        step_logits_list: list[torch.Tensor],
        step_exit_logits_list: list[torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        t_steps = len(step_logits_list)
        assert t_steps == len(step_exit_logits_list), "Mismatch in recurrent steps."

        valid_mask = self._valid_mask(labels)  # [B, S]

        per_step_ce = []
        for logits in step_logits_list:
            ce = token_cross_entropy_per_token(
                logits, labels, ignore_index=self.ignore_index
            ).detach()
            per_step_ce.append(ce)
        per_step_ce = torch.stack(per_step_ce, dim=0)  # [T, B, S]

        lambdas, _, _ = build_exit_distribution(step_exit_logits_list)
        continuation_probs = 1.0 - lambdas

        adaptive_sums = []
        improvements = []
        targets = []

        eps = 1e-8

        for t in range(1, t_steps):
            improve_t = torch.clamp(per_step_ce[t - 1] - per_step_ce[t], min=0.0)
            improvements.append(improve_t)

            w_t = torch.sigmoid(self.k_improve * (improve_t - self.gamma_improve))
            targets.append(w_t)

            pred_continue = continuation_probs[t].clamp(min=eps, max=1.0 - eps)
            pred_exit = lambdas[t].clamp(min=eps, max=1.0 - eps)

            bce_t = -(
                w_t * torch.log(pred_continue) + (1.0 - w_t) * torch.log(pred_exit)
            )
            adaptive_sums.append((bce_t * valid_mask).sum())

        if len(adaptive_sums) == 0:
            z = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
            return {
                "loss_sum": z,
                "loss": z,
                "mean_improvement": z,
                "mean_target_continue": z,
                "lambdas": lambdas.detach(),
            }

        denom_t = max(t_steps - 1, 1)
        loss_sum = torch.stack(adaptive_sums).sum() / denom_t
        loss_mean = loss_sum / valid_mask.sum().clamp_min(eps)

        mean_improvement = reduce_masked_mean(
            torch.stack(improvements).mean(dim=0), valid_mask
        )
        mean_target_continue = reduce_masked_mean(
            torch.stack(targets).mean(dim=0), valid_mask
        )

        return {
            "loss_sum": loss_sum,
            "loss": loss_mean,
            "mean_improvement": mean_improvement,
            "mean_target_continue": mean_target_continue,
            "lambdas": lambdas.detach(),
        }


def build_ouro_loop_lm_loss(compile_config: CompileConfig, **kwargs):
    del kwargs
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Ouro loop LM loss is not torch.compiled (dynamic objective).")

    def loss_fn(
        pred: torch.Tensor,
        labels: torch.Tensor,
        *,
        model: nn.Module | None = None,
        **kw: object,
    ) -> torch.Tensor:
        del kw
        if model is None:
            return cross_entropy_loss(pred, labels)

        from torchtitan.models.ouro.model import OuroModel

        ouro: OuroModel | None = None
        for m in model.modules():
            if isinstance(m, OuroModel):
                ouro = m
                break

        if ouro is None or ouro.loop_lm_loss == "none":
            return cross_entropy_loss(pred, labels)

        cache = getattr(ouro, "_ouro_loop_lm_cache", None)
        if cache is None:
            return cross_entropy_loss(pred, labels)

        step_logits_list, step_exit_logits_list = cache
        delattr(ouro, "_ouro_loop_lm_cache")
        assert ouro._loop_lm is not None
        if ouro.loop_lm_loss == "stage1":
            return ouro._loop_lm.stage1_loss(
                step_logits_list, step_exit_logits_list, labels
            )["loss_sum"]
        if ouro.loop_lm_loss == "stage2":
            return ouro._loop_lm.stage2_loss(
                step_logits_list, step_exit_logits_list, labels
            )["loss_sum"]
        return cross_entropy_loss(pred, labels)

    return loss_fn
