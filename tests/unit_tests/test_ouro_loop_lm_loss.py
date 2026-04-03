# SPDX-License-Identifier: Apache-2.0

import torch

from torchtitan.components.loss import (
    IGNORE_INDEX,
    LoopLMLoss,
    build_exit_distribution,
    token_cross_entropy_per_token,
)


def test_build_exit_distribution_masses_sum_to_one():
    b, s, t = 2, 5, 4
    exit_logits = [torch.randn(b, s) for _ in range(t)]
    _, p_exit, _ = build_exit_distribution(exit_logits)
    mass = p_exit.sum(dim=0)
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5, rtol=1e-5)


def test_token_cross_entropy_ignore():
    b, s, v = 2, 4, 8
    logits = torch.randn(b, s, v)
    labels = torch.full((b, s), IGNORE_INDEX, dtype=torch.long)
    labels[:, 0] = 0
    ce = token_cross_entropy_per_token(logits, labels, ignore_index=IGNORE_INDEX)
    assert ce[:, 1:].abs().sum() == 0


def test_loop_lm_stage1_finite():
    t, b, s, v = 3, 2, 4, 16
    labels = torch.randint(0, v, (b, s))
    step_logits = [torch.randn(b, s, v) for _ in range(t)]
    step_exit = [torch.randn(b, s, 1) for _ in range(t)]
    loss_mod = LoopLMLoss(ignore_index=IGNORE_INDEX)
    out = loss_mod.stage1_loss(step_logits, step_exit, labels)
    assert out["loss_sum"].shape == ()
    assert torch.isfinite(out["loss_sum"])


def test_loop_lm_stage2_finite():
    t, b, s, v = 3, 2, 4, 16
    labels = torch.randint(0, v, (b, s))
    step_logits = [torch.randn(b, s, v) for _ in range(t)]
    step_exit = [torch.randn(b, s, 1) for _ in range(t)]
    loss_mod = LoopLMLoss(ignore_index=IGNORE_INDEX)
    out = loss_mod.stage2_loss(step_logits, step_exit, labels)
    assert out["loss_sum"].shape == ()
    assert torch.isfinite(out["loss_sum"])
