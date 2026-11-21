"""Stage-II training of a candidate router on cached states (fast router NAS; features from nas_feats.py).

build_router is the construction of torchtitan/models/ouro/model.py::build_router (the Tier-1 axes of boptim-agent's
objective/ouro_router_nas.py), and the loss is torchtitan/components/loss.py::loop_lm_adaptive_gate_loss_sum:

    improvement_t = relu(L_{t-1} - L_t)                      L_t = cross-entropy of the readout after loop t+1 (0-based t)
    w_t           = sigmoid(k * (improvement_t - gamma))     ideal continuation label
    loss          = sum over t in {1, 2} of BCE(w_t, 1 - lambda_t) / T,   lambda_t = sigmoid(router(state after loop t+1))

Only the gates after loops 2 and 3 get a label (the gate after loop 4 is never used, the gate after loop 1 has no "previous
loop" to compare with); the router is one module shared by all loops, so its output after loop 1 is what it generalises to.
The language model is frozen in Stage II, so the cached states are exactly the router's training inputs.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import nn

ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
N_LOOPS = 4


def build_router(dim: int, spec: dict) -> nn.Module:
    layers = int(spec.get("layers", 0))
    hidden = int(spec.get("hidden", 128))
    act = str(spec.get("act", "gelu")).lower()
    norm = bool(spec.get("norm", False))
    if layers <= 0 and not norm:
        return nn.Linear(dim, 1, bias=True)
    mods: list[nn.Module] = []
    if norm:
        mods.append(nn.LayerNorm(dim))
    d = dim
    for _ in range(layers):
        mods.append(nn.Linear(d, hidden))
        mods.append(ACTS[act]())
        d = hidden
    mods.append(nn.Linear(d, 1))
    return nn.Sequential(*mods)


def describe_arch(spec: dict) -> str:
    if int(spec.get("layers", 0)) == 0:
        return "linear-ln" if spec.get("norm") else "linear"
    return f"L{spec['layers']}-H{spec['hidden']}-{spec['act']}-{'ln' if spec.get('norm') else 'no-ln'}"


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def fit_router(feats: dict, spec: dict, gamma: float, *, k: float = 50.0, steps: int = 300, lr: float = 1e-3, bs: int = 8192,
               seed: int = 0, device: str = "cuda"):
    """Train a router from a random initialisation. Returns (router in float32 on `device`, diagnostics)."""
    t0 = time.time()
    h, ce = feats["h"], feats["ce"]                     # (N, 3, D) float16 on the CPU, (N, 4) float32
    n, _, dim = h.shape
    torch.manual_seed(int(seed))
    router = build_router(dim, spec).to(device).float()
    w_all = torch.sigmoid(float(k) * (torch.relu(ce[:, :-1] - ce[:, 1:]) - float(gamma)))    # (N, 3): column t-1 labels gate t
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(int(seed))
    eps = 1e-7
    last = None
    router.train()
    for step in range(int(steps)):
        for g in opt.param_groups:
            g["lr"] = lr * 0.5 * (1.0 + math.cos(math.pi * step / max(1, int(steps))))
        idx = torch.randint(0, n, (bs,), generator=gen)
        x = h[idx, 1:3].to(device, non_blocking=True).float()                   # states after loops 2 and 3: (bs, 2, D)
        w = w_all[idx, 0:2].to(device, non_blocking=True)                        # labels of the gates after loops 2 and 3
        lam = torch.sigmoid(router(x).squeeze(-1)).clamp(eps, 1.0 - eps)
        loss = -(w * (1.0 - lam).log() + (1.0 - w) * lam.log()).sum() / bs / float(N_LOOPS)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    router.eval()
    diag = {"fit_loss": last, "fit_s": time.time() - t0, "n_params": n_params(router), "label_mean": [float(w_all[:, 0].mean()), float(w_all[:, 1].mean())]}
    with torch.no_grad():                                 # what the trained router says after every loop, on a fixed sample
        sub = torch.arange(0, n, max(1, n // 4096))[:4096]
        lam = torch.sigmoid(router(h[sub].to(device).float()).squeeze(-1))    # (m, 3)
        diag["lambda_mean"] = [float(v) for v in lam.mean(0)]
    return router, diag


def save_router(path: str, router: nn.Module, spec: dict, **meta) -> None:
    """File read by the gate-lab serving model (boptim_lcb/make_gatelab_modeling.py, option decode_router=<path>)."""
    sd = {k: v.detach().float().cpu() for k, v in router.state_dict().items()}
    torch.save(dict(meta, spec=dict(spec), state_dict=sd), path)


def released_gate_router(model_file: str = "/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking/model.safetensors"):
    """The released gate (nn.Linear(2048, 1)) as a router module, for checks of the gate-lab path."""
    from safetensors import safe_open

    with safe_open(model_file, framework="pt") as f:
        keys = [k for k in f.keys() if k.endswith("early_exit_gate.weight") or k.endswith("early_exit_gate.bias")]
        w = {k.rsplit(".", 1)[1]: f.get_tensor(k) for k in keys}
    lin = nn.Linear(w["weight"].shape[1], 1)
    with torch.no_grad():
        lin.weight.copy_(w["weight"].float())
        lin.bias.copy_(w["bias"].float())
    return lin


def load_feats(path: str) -> dict:
    f = torch.load(path, map_location="cpu")
    return {"h": f["h"], "ce": f["ce"].float(), "ok": f["ok"], "gate": f.get("gate")}
