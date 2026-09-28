#!/usr/bin/env python3
"""Diagnostic: does the Ouro early-exit gate ever trigger, and does SFT change it?

Builds the Ouro 1.4B model, loads the HF base backbone+gate, runs a forward over
a sample prompt computing the gate at every UT step, and reports the per-token
exit-step distribution for several thresholds. Repeats with an SFT gate overlaid
from a DCP checkpoint so we can see whether SFT shifts exit behaviour.

Run (CPU is fine, ~1.4B):
  ./.venv/bin/python scripts/diag_early_exit.py \
      --sft_checkpoint /work/nvme/bdjz/jli37/boptim/eval001_lr1.00e-02_20260619_013915/checkpoint/step-100
"""
import argparse
import copy
import dataclasses

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger

PROMPT = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    """
'''


def build_model(device):
    cm = ConfigManager()
    config = cm.parse_args(["--module", "ouro", "--config", "ouro_1_4b"])
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    tok = HuggingFaceTokenizer.Config().build(tokenizer_path=config.hf_assets_path)
    mc = copy.deepcopy(config.model_spec.model)
    mc.update_from_config(trainer_config=config)
    # keep threshold=1.0 so forward() runs ALL ut steps; we compute exits manually
    mc = dataclasses.replace(mc, early_exit_threshold=1.0)
    with torch.device(device):
        model = mc.build()
    model.to(device).eval()
    with torch.no_grad():
        model.init_weights()
    # load HF base (backbone + pretrained gate)
    sd_adapter = config.model_spec.state_dict_adapter(mc, config.hf_assets_path)
    from torch.distributed.checkpoint import HuggingFaceStorageReader
    sd = model.state_dict()
    hf_sd = sd_adapter.to_hf(sd)
    dcp.load(hf_sd, storage_reader=HuggingFaceStorageReader(path=config.hf_assets_path))
    model.load_state_dict(sd_adapter.from_hf(hf_sd), strict=True)
    return model, tok


@torch.no_grad()
def gate_lambdas(model, tokens):
    """Return lambda_i = sigmoid(gate(h_i)) for each UT step: shape [steps, seq]."""
    h = model.tok_embeddings(tokens)
    lambdas = []
    for _ in range(model.total_ut_steps):
        for layer in model.layers.values():
            h = layer(h, model.freqs_cis, None, None)
        h = model.norm(h)
        g = model.early_exit_gate(h).squeeze(-1).float()  # [b, seq]
        lambdas.append(torch.sigmoid(g)[0])               # [seq]
    return torch.stack(lambdas, dim=0)  # [steps, seq]


def exit_steps(lambdas, threshold):
    """Stick-breaking exit PDF -> per-token exit step at *threshold* (mirror model)."""
    steps, seq = lambdas.shape
    remaining = torch.ones(seq)
    cum = torch.zeros(seq)
    exited = torch.zeros(seq, dtype=torch.bool)
    out = torch.full((seq,), steps - 1, dtype=torch.long)
    for i in range(steps):
        lam = lambdas[i]
        is_last = i == steps - 1
        p = remaining if is_last else lam * remaining
        remaining = remaining * (1.0 - lam)
        cum = cum + p
        newly = (~exited) & ((cum >= threshold) | is_last)
        out[newly] = i
        exited = exited | newly
    return out


def report(tag, lambdas, total_steps):
    print(f"\n=== {tag} ===")
    print("mean lambda per UT step:", [round(float(lambdas[i].mean()), 4) for i in range(total_steps)])
    for thr in (0.5, 0.7, 0.9, 0.99):
        es = exit_steps(lambdas, thr)
        hist = [int((es == s).sum()) for s in range(total_steps)]
        print(f"  thr={thr:<4}  exit-step hist {hist}  mean_exit={float(es.float().mean()):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    init_logger()

    model, tok = build_model(args.device)
    ids = torch.tensor(tok.encode(PROMPT, add_bos=False, add_eos=False),
                       dtype=torch.long, device=args.device).unsqueeze(0)
    print(f"prompt tokens: {ids.shape[1]}")

    base_lam = gate_lambdas(model, ids)
    report("BASE gate", base_lam, model.total_ut_steps)

    # overlay SFT gate
    gsd = {"early_exit_gate.weight": torch.zeros(1, 2048),
           "early_exit_gate.bias": torch.zeros(1)}
    dcp.load(gsd, checkpoint_id=args.sft_checkpoint)
    model.early_exit_gate.weight.copy_(gsd["early_exit_gate.weight"].to(model.early_exit_gate.weight.dtype))
    model.early_exit_gate.bias.copy_(gsd["early_exit_gate.bias"].to(model.early_exit_gate.bias.dtype))
    sft_lam = gate_lambdas(model, ids)
    report("SFT gate", sft_lam, model.total_ut_steps)


if __name__ == "__main__":
    main()
