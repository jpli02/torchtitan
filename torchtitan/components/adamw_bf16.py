"""AdamW with both moment buffers stored in bfloat16.

Why it exists: full fine-tuning of Ouro-2.6B does not fit one 46 GB card with fp32 AdamW (two fp32 moment buffers = 21 GB),
and the 8-bit AdamW used instead (torchao AdamW8bit) stores each moment RELATIVE to the largest value in its 256-element
block on a 256-level grid. A second moment below about 3e-7 of its block's maximum is stored as exactly zero while the first
moment (which scales like the gradient, not its square) survives, so parameters whose gradients are small or rare next to a
neighbour's take steps far larger than the learning rate (see lr_debug_8bit.py and the optimizer probe).

bfloat16 keeps an exponent PER ELEMENT, so a small moment keeps its own scale (8 significant bits, range 1e-38..3e38): 2 bytes
per moment instead of 1, i.e. about 10.7 GB for 2.6B parameters instead of 5.3 GB (fp32: 21.4 GB). The arithmetic is done in
fp32 from the stored values, the parameter update uses the fp32 moments of the current step (as torchao does), and the moments
are written back with unbiased stochastic rounding, so the exponential averages are not biased by the 8-bit mantissa.
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


def _fp32_to_bf16_sr(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 with stochastic rounding: round up with probability equal to the discarded fraction (unbiased)."""
    bits = x.view(torch.int32)
    frac = bits & 0xFFFF
    rnd = torch.randint_like(bits, 0, 1 << 16)
    bits = torch.where(rnd < frac, bits + 0x10000, bits) & ~0xFFFF     # magnitude grows in both signs: sign-magnitude format
    return bits.view(torch.float32).to(torch.bfloat16)


class AdamWBF16(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, stochastic_round=True):
        if lr < 0 or eps < 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1 or weight_decay < 0:
            raise ValueError("invalid AdamWBF16 hyper-parameter")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, stochastic_round=stochastic_round))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            b1, b2 = group["betas"]
            eps, wd, sr = group["eps"], group["weight_decay"], group["stochastic_round"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamWBF16 does not support sparse gradients")
                # under FSDP a parameter is a DTensor: work on the local shard, like torchao's low-bit optimizers
                pl = getattr(p, "_local_tensor", p)
                g = getattr(p.grad, "_local_tensor", p.grad).float()
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(pl, dtype=torch.bfloat16)
                    state["exp_avg_sq"] = torch.zeros_like(pl, dtype=torch.bfloat16)
                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"].float().lerp_(g, 1 - b1)
                v = state["exp_avg_sq"].float().lerp_(g * g, 1 - b2)
                state["exp_avg"].copy_(_fp32_to_bf16_sr(m) if sr else m)
                state["exp_avg_sq"].copy_(_fp32_to_bf16_sr(v) if sr else v)
                denom = v.sqrt_().div_(math.sqrt(1 - b2 ** t)).add_(eps)
                if pl.dtype == torch.float32:
                    if wd:
                        pl.mul_(1 - lr * wd)
                    pl.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))
                else:
                    pf = pl.float()
                    if wd:
                        pf.mul_(1 - lr * wd)
                    pf.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))
                    pl.copy_(pf)
        return loss
