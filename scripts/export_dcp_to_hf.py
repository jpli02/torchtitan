#!/usr/bin/env python3
"""Export a trained TorchTitan Ouro checkpoint to a HuggingFace model directory.

Bridges the "checkpoint transfer" gap: boptim trains -> DCP checkpoint, but the
fast KV-cache eval path (evaluate_humaneval_evalplus.py --kv_cache) loads HF
safetensors via modeling_ouro.py. This loads the DCP (or HF) weights into the
TorchTitan OuroModel, maps them to HF names with OuroStateDictAdapter.to_hf, writes
model.safetensors, and copies the static HF assets (config.json, tokenizer,
modeling_ouro.py, ...) from a template HF dir so the result is a self-contained,
from_pretrained-loadable checkpoint.

Usage (trained checkpoint):
    python scripts/export_dcp_to_hf.py --module ouro --config ouro_1_4b \
        --checkpoint <DCP_dir> --hf_template ./assets/hf/Ouro-1.4B --out <OUT_dir>

Round-trip self-test (no trained checkpoint needed): pass --hf_checkpoint instead
of --checkpoint to load the released weights and re-export them; --verify then
asserts to_hf(from_hf(w)) == w against the template's model.safetensors.
"""
import argparse
import copy
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
from evaluate_humaneval import _load_model  # noqa: E402
from torchtitan.config import ConfigManager  # noqa: E402

# Static HF assets to copy verbatim from the template dir (everything the model
# needs at from_pretrained time except the weights we regenerate).
_SKIP = {"model.safetensors", "model.safetensors.index.json", ".cache", ".git"}


def _config_and_adapter(module: str, config_name: str, hf_template: str | None):
    """Parse the torchtitan config, returning (hf_assets_path, adapter). When
    hf_template is None it defaults to the config's own hf_assets_path (the
    released HF dir the run was initialized from)."""
    config = ConfigManager().parse_args(["--module", module, "--config", config_name])
    template = hf_template or config.hf_assets_path
    model_config = copy.deepcopy(config.model_spec.model)
    model_config.update_from_config(trainer_config=config)
    adapter = config.model_spec.state_dict_adapter(model_config, template)
    if adapter is None:
        raise RuntimeError("Ouro model spec provides no state_dict_adapter.")
    return template, adapter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="ouro")
    ap.add_argument("--config", default="ouro_1_4b")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="trained DCP checkpoint dir")
    src.add_argument("--hf_checkpoint", help="HF dir (round-trip self-test)")
    ap.add_argument("--hf_template", default=None,
                    help="released HF dir supplying config/tokenizer/modeling code "
                         "(default: the config's hf_assets_path)")
    ap.add_argument("--out", required=True, help="output HF checkpoint dir")
    ap.add_argument("--dtype", choices=["bf16", "fp32", "native"], default="native",
                    help="on-disk weight dtype (default: keep model's dtype)")
    ap.add_argument("--verify", action="store_true",
                    help="assert exported weights match --hf_template (round-trip)")
    args = ap.parse_args()

    from safetensors.torch import save_file, load_file

    # 1) Load weights into the TorchTitan OuroModel (DCP or HF path).
    hf_template, adapter = _config_and_adapter(
        args.module, args.config, args.hf_template)
    model, _tok = _load_model(args.module, args.config, args.checkpoint,
                              args.hf_checkpoint)

    # 2) TorchTitan -> HF names, on CPU + contiguous for safetensors.
    cast = {"bf16": torch.bfloat16, "fp32": torch.float32}.get(args.dtype)
    hf_sd = {}
    for k, v in adapter.to_hf(model.state_dict()).items():
        v = v.detach().cpu().contiguous()
        if cast is not None:
            v = v.to(cast)
        hf_sd[k] = v
    if not hf_sd:
        raise RuntimeError("to_hf produced an empty state dict -- adapter mismatch.")

    # 3) Copy static assets, then write weights.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(hf_template).iterdir():
        if f.name in _SKIP:
            continue
        dst = out / f.name
        if f.is_dir():
            shutil.copytree(f, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(f, dst)
    save_file(hf_sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"Exported {len(hf_sd)} tensors -> {out/'model.safetensors'}")
    print(f"Copied HF assets from {hf_template} (config/tokenizer/modeling)")

    # 4) Optional round-trip check against the template weights.
    if args.verify:
        ref = load_file(str(Path(hf_template) / "model.safetensors"))
        miss = set(ref) ^ set(hf_sd)
        assert not miss, f"key mismatch vs template: {sorted(miss)[:6]}"
        bad = []
        for k in ref:
            a, b = hf_sd[k].float(), ref[k].float()
            if a.shape != b.shape or not torch.equal(a, b):
                bad.append((k, float((a - b).abs().max()) if a.shape == b.shape else -1))
        if bad:
            print(f"VERIFY FAIL: {len(bad)} tensors differ, e.g. {bad[:5]}")
            sys.exit(1)
        print(f"VERIFY OK: all {len(ref)} tensors bit-identical to template "
              f"(to_hf(from_hf(w)) == w)")


if __name__ == "__main__":
    main()
