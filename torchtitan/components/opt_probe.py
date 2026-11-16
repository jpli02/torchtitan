"""Optimizer probe (debug only; active when OURO_OPT_PROBE=<out.jsonl> is set).

Question it answers: does the optimizer in use (torchao AdamW8bit: moments quantised to 8 bits relative to the largest value in
each 256-element block) take the steps that exact AdamW would take for the same gradients? For a sample of parameter tensors it
keeps an exact fp32 AdamW in the shadow (same lr, betas, eps, weight decay, same gradient history) and compares, at every
optimizer step, the update that was really applied (parameter after minus before) with the update exact AdamW would apply.

Per step and tensor it writes: the norms of both updates, their cosine, the share of elements whose real step exceeds 1.5 x lr
(exact AdamW with these betas cannot exceed about 1 x lr per element once bias correction has settled) and the largest
real step in units of lr; and the same for the displacement accumulated since step 0.

Only slices of at most OURO_OPT_PROBE_ELEMS elements (default 2**20, block aligned from the start of the tensor) are tracked,
so the cost is a few hundred MB and a few ms per step.
"""
from __future__ import annotations

import json
import os

import torch


def _local(x):
    """The plain local tensor of a parameter or gradient (a DTensor under FSDP wraps one)."""
    return getattr(x, "_local_tensor", x)


class OptProbe:
    def __init__(self, container, out_path: str):
        self.out_path = out_path
        self.container = container
        self.max_elems = int(os.environ.get("OURO_OPT_PROBE_ELEMS", str(1 << 20)))
        self.max_tensors = int(os.environ.get("OURO_OPT_PROBE_TENSORS", "14"))
        self.t = 0
        self.items = []
        named = []
        for model in container.model_parts:
            named += [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        big = [(n, p) for n, p in named if p.numel() >= 4096 and p.numel() % 256 == 0]     # the ones torchao quantises
        small = [(n, p) for n, p in named if not (p.numel() >= 4096 and p.numel() % 256 == 0)]
        picks = []
        if big:
            emb = max(big, key=lambda x: x[1].numel())                                       # the embedding / output matrix
            picks.append(emb)
            rest = [x for x in big if x[0] != emb[0]]
            k = max(1, self.max_tensors - 3)
            step = max(1, len(rest) // k)
            picks += rest[::step][:k]
        picks += small[:2]                                                                  # fp32 states in torchao: a control
        for n, p in picks:
            m = min(p.numel(), self.max_elems)
            self.items.append({"name": n, "p": p, "n": m, "quantised": p.numel() >= 4096 and p.numel() % 256 == 0,
                               "m": torch.zeros(m, device=p.device, dtype=torch.float32),
                               "v": torch.zeros(m, device=p.device, dtype=torch.float32),
                               "d_real": torch.zeros(m, device=p.device, dtype=torch.float32),
                               "d_exact": torch.zeros(m, device=p.device, dtype=torch.float32)})
        with open(self.out_path, "w") as f:
            f.write(json.dumps({"probe": "start", "tensors": [{"name": it["name"], "numel": it["p"].numel(), "tracked": it["n"],
                                                               "quantised": it["quantised"]} for it in self.items]}) + "\n")

    @torch.no_grad()
    def before(self):
        g0 = self.container.optimizers[0].param_groups[0]
        self.lr = float(g0["lr"])
        self.betas, self.eps, self.wd = g0["betas"], float(g0["eps"]), float(g0["weight_decay"])
        for it in self.items:
            p = it["p"]
            it["has_grad"] = p.grad is not None
            if not it["has_grad"]:
                continue
            it["p0"] = _local(p.detach()).reshape(-1)[: it["n"]].float().clone()
            it["g"] = _local(p.grad.detach()).reshape(-1)[: it["n"]].float().clone()

    @torch.no_grad()
    def after(self):
        self.t += 1
        b1, b2 = self.betas
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        rows = []
        for it in self.items:
            if not it.get("has_grad"):
                continue
            p1 = _local(it["p"].detach()).reshape(-1)[: it["n"]].float()
            u_real = p1 - it["p0"]
            g = it["g"]
            it["m"].lerp_(g, 1 - b1)
            it["v"].lerp_(g * g, 1 - b2)
            u_exact = -self.lr * self.wd * it["p0"] - self.lr * (it["m"] / bc1) / (it["v"].sqrt() / (bc2 ** 0.5) + self.eps)
            it["d_real"] += u_real
            it["d_exact"] += u_exact
            a = u_real.abs()

            def cos(x, y):
                return float(torch.nn.functional.cosine_similarity(x, y, dim=0))

            rows.append({"step": self.t, "lr": self.lr, "name": it["name"],
                         "g_absmax": float(g.abs().max()), "g_rms": float(g.pow(2).mean().sqrt()),
                         "g_zero_share": float((g == 0).float().mean()),
                         "u_real": float(u_real.norm()), "u_exact": float(u_exact.norm()), "u_cos": cos(u_real, u_exact),
                         "share_over_1p5lr": float((a > 1.5 * self.lr).float().mean()),
                         "share_over_10lr": float((a > 10 * self.lr).float().mean()),
                         "max_step_in_lr": float(a.max() / max(self.lr, 1e-30)),
                         "max_exact_in_lr": float(u_exact.abs().max() / max(self.lr, 1e-30)),
                         "d_real": float(it["d_real"].norm()), "d_exact": float(it["d_exact"].norm()),
                         "d_cos": cos(it["d_real"], it["d_exact"]),
                         "p_rms": float(it["p0"].pow(2).mean().sqrt())})
            it.pop("p0", None)
            it.pop("g", None)
        with open(self.out_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def get_probe(container):
    """The container's probe, built on first use; None unless OURO_OPT_PROBE is set."""
    path = os.environ.get("OURO_OPT_PROBE")
    if not path:
        return None
    probe = getattr(container, "_ouro_probe", None)
    if probe is None:
        probe = OptProbe(container, path)
        container._ouro_probe = probe
    return probe
