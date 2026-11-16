"""Learning-rate debug, part 3: does the 8-bit AdamW (torchao, quantised moments) take the same steps as 32-bit AdamW when the
gradients look like fine-tuning at batch 1: noisy, with very different scales inside one quantisation block, and rare for some
parameters? A mis-quantised second moment would make the EFFECTIVE learning rate of some parameters much larger than the nominal
one, which a constant-gradient check cannot see.

Both optimizers get identical gradients for N steps with the recipe's settings (lr 2e-5 constant, betas 0.9/0.95, eps 1e-8, no
weight decay); reported: the per-parameter displacement of 8-bit relative to 32-bit.

  CUDA_VISIBLE_DEVICES="" TORCHDYNAMO_DISABLE=1 /home/jli199/torchtitan/.venv/bin/python lr_debug_8bit.py
"""
import torch
from torchao.optim import AdamW8bit

torch.manual_seed(0)
N, STEPS, LR = 1 << 16, 400, 2e-5
kw = dict(lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)


import sys

sys.path.insert(0, ".")
WHICH = sys.argv[1] if len(sys.argv) > 1 else "8bit"      # "8bit" (torchao AdamW8bit) or "bf16" (torchtitan AdamWBF16)


def run(case, scale, p_active, mean_frac):
    w8 = torch.nn.Parameter(torch.zeros(N))
    w32 = torch.nn.Parameter(torch.zeros(N))
    if WHICH == "bf16":
        from torchtitan.components.adamw_bf16 import AdamWBF16
        o8 = AdamWBF16([w8], **kw)
    else:
        o8 = AdamW8bit([w8], **kw)
    o32 = torch.optim.AdamW([w32], **kw)
    g = torch.Generator().manual_seed(1)
    sign = torch.where(torch.rand(N, generator=g) < 0.5, -1.0, 1.0)
    for _ in range(STEPS):
        active = (torch.rand(N, generator=g) < p_active).float()
        grad = scale * active * (torch.randn(N, generator=g) + mean_frac * sign)
        for w, o in ((w8, o8), (w32, o32)):
            o.zero_grad()
            w.grad = grad.clone()
            o.step()
    d8, d32 = w8.detach(), w32.detach()
    ratio = (d8.abs() + 1e-12) / (d32.abs() + 1e-12)
    q = torch.quantile(ratio, torch.tensor([0.01, 0.5, 0.99]))
    cos = float(torch.nn.functional.cosine_similarity(d8, d32, dim=0))
    print(f"{case:44s} |d8|/|d32| norm {float(d8.norm() / d32.norm()):.3f}  cosine {cos:.4f}  per-param ratio p1 {q[0]:.2f} median {q[1]:.2f} "
          f"p99 {q[2]:.2f} max {float(ratio.max()):.1f}  mean |d32| / (lr*steps) {float(d32.abs().mean()) / (LR * STEPS):.3f}")
    # the same, split by gradient scale (are the small-gradient parameters of a block the ones that go wrong?)
    if scale.numel() == N and float(scale.max() / scale.min()) > 10:
        order = torch.argsort(scale)
        for name, idx in (("smallest-scale tenth", order[: N // 10]), ("largest-scale tenth", order[-N // 10:])):
            r = ratio[idx]
            print(f"{'':6s}{name:22s} median ratio {float(r.median()):.2f}  p99 {float(torch.quantile(r, 0.99)):.2f}  "
                  f"mean |d8| {float(d8[idx].abs().mean()):.2e}  mean |d32| {float(d32[idx].abs().mean()):.2e}")


ones = torch.ones(N)
logu = 10 ** (-6 * torch.rand(N, generator=torch.Generator().manual_seed(2)))          # scales from 1e-6 to 1 inside every block
print(f"optimizer under test: {WHICH} (columns named d8 are this optimizer, d32 is torch AdamW with fp32 moments)")
print(f"{STEPS} steps at lr {LR}; a parameter with a perfectly consistent gradient moves lr*steps = {LR * STEPS:.1e}")
run("dense noise, equal scales", ones, 1.0, 0.0)
run("dense noise + 20 % consistent part", ones, 1.0, 0.2)
run("dense noise, scales 1e-6..1 in a block", logu, 1.0, 0.0)
run("scales 1e-6..1, 20 % consistent part", logu, 1.0, 0.2)
run("rare gradients (5 % of steps), equal scales", ones, 0.05, 0.0)
run("rare gradients (5 %), scales 1e-6..1", logu, 0.05, 0.2)
