"""
scripts/extract_router.py
--------------------------
Extract just the trained early_exit_gate (router) weights from a full Ouro
SFT checkpoint and save them as a tiny standalone artifact.

Motivation: stage2_adaptive SFT freezes the entire 1.4B-parameter backbone and
only trains the router (2049 params for the original nn.Linear(2048,1) gate,
or a few hundred KB for a small NAS MLP router). Saving/uploading the full
model every time an eval finishes is enormously wasteful when the only thing
that actually changed is ~2KB of weights -- and it's exactly what made the
earlier "best" GPT/Qwen checkpoints unrecoverable once disk cleanup removed
the full-model saves. This script pulls out just the router.

Usage:
    .venv/bin/python scripts/extract_router.py \
        --checkpoint /path/to/step-100 \
        --output router_gamma1.0_thr0.5.safetensors \
        --meta '{"adaptive_gamma": 1.0, "early_exit_threshold": 0.5, "avg_loops": 1.2077, "pass@1": 0.7073}'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def extract_router_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Pull early_exit_gate.* tensors out of an HF-format (safetensors) Ouro
    checkpoint directory. Handles both a single model.safetensors / model-*-of-*
    shard (last_save_in_hf output) and a sharded-with-index checkpoint."""
    shard_files = sorted(checkpoint_dir.glob("model*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No *.safetensors files found under {checkpoint_dir}")

    router_sd: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        sd = load_file(str(shard))
        for k, v in sd.items():
            # HF key prefix is "model.early_exit_gate...."; torchtitan's own
            # checkpoint format (no prefix) is "early_exit_gate....". Normalize
            # to the un-prefixed torchtitan name in the extracted artifact.
            if "early_exit_gate" in k:
                short_key = k.split("early_exit_gate", 1)
                router_sd["early_exit_gate" + short_key[1]] = v.clone()
    if not router_sd:
        raise ValueError(f"No early_exit_gate.* keys found in {checkpoint_dir} "
                          f"(looked in {[str(f) for f in shard_files]})")
    return router_sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path,
                     help="HF-format checkpoint dir (contains model*.safetensors)")
    ap.add_argument("--output", required=True, type=Path,
                     help="Output .safetensors path for the extracted router")
    ap.add_argument("--meta", default="{}",
                     help="JSON string of metadata to embed (gamma, threshold, "
                          "avg_loops, pass@1, router_spec, etc.)")
    args = ap.parse_args()

    router_sd = extract_router_state_dict(args.checkpoint)
    total_bytes = sum(v.numel() * v.element_size() for v in router_sd.values())
    print(f"Extracted {len(router_sd)} tensor(s), "
          f"{sum(v.numel() for v in router_sd.values())} params, "
          f"{total_bytes} bytes:")
    for k, v in router_sd.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    meta = json.loads(args.meta)
    # safetensors metadata values must be strings.
    str_meta = {k: json.dumps(v) if not isinstance(v, str) else v
                for k, v in meta.items()}
    str_meta["source_checkpoint"] = str(args.checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(router_sd, str(args.output), metadata=str_meta)
    print(f"Saved -> {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
