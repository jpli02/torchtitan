# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import TypeAlias

import torch

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

def ouro_adaptive_loss(
    step_preds: torch.Tensor, 
    exit_probs: torch.Tensor, 
    labels: torch.Tensor,
    beta: float = 0.05,
    lambda_: float = 0.5,
) -> torch.Tensor:
    """
    Ouro adaptive computation loss with a tunable Geometric Prior.
    
    Args:
        step_preds: Tensor of shape (batch, seq, num_steps, vocab_size)
        exit_probs: Tensor of shape (batch, seq, num_steps) containing p_phi
        labels: Tensor of shape (batch, seq)
        beta: Entropy regularization coefficient
        lambda_: The geometric prior hyperparameter 
    """
    num_steps = step_preds.size(2)
    device = step_preds.device
    
    step_preds_flat = step_preds.flatten(0, 2).float()
    labels_rep = labels.unsqueeze(-1).expand(-1, -1, num_steps).flatten()

    ce_loss_flat = cross_entropy_loss(step_preds_flat, labels_rep)
    ce_loss = ce_loss_flat.view(*labels.shape, num_steps)
    
    expected_task_loss = (exit_probs * ce_loss).sum(dim=-1)
    # pi_lambda(t) = lambda * (1-lambda)^(t) for 0-indexed t
    steps = torch.arange(num_steps, device=device)
    prior_probs = lambda_ * (1.0 - lambda_) ** steps
    prior_probs[-1] = 1.0 - prior_probs[:-1].sum()
    
    prior_probs = prior_probs.view(1, 1, num_steps)

    # KL(p || pi) = Sum p * log(p / pi)
    eps = 1e-8
    kl_div = (exit_probs * torch.log((exit_probs + eps) / (prior_probs + eps))).sum(dim=-1)

    # Masking and Reduction
    valid_mask = (labels != IGNORE_INDEX).float()
    
    total_token_loss = expected_task_loss + beta * kl_div
    
    return (total_token_loss * valid_mask).sum()


def build_ouro_adaptive_loss(compile_config: CompileConfig, **kwargs):
    """Builder for the Ouro geometric prior loss."""
    beta = kwargs.get("beta", 0.05)
    lambda_ = kwargs.get("lambda_", 0.5)
    
    def loss_fn(step_preds, exit_probs, labels):
        return ouro_adaptive_loss(step_preds, exit_probs, labels, beta=beta, lambda_=lambda_)
    
    if compile_config.enable and "loss" in compile_config.components:
        logger.info(f"Compiling the Ouro adaptive loss function (beta={beta}, lambda={lambda_}) with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
        
    return loss_fn