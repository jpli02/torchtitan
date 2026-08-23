"""
scripts/push_ouro_sft_to_hub.py
--------------------------------
Push a finished Ouro terminal-agent SFT checkpoint to the Hugging Face Hub.

torchtitan's `checkpoint.last_save_in_hf=True` writes the final step as HF
safetensors, but a servable/shareable repo needs more than the weights: Ouro is
a custom architecture loaded through `trust_remote_code`, so the modeling and
configuration modules plus the tokenizer must travel with it or the repo cannot
be loaded by anyone (including our own scripts/ouro_openai_server.py). This
script copies those companion files from the base checkpoint the run started
from, then uploads the whole directory.

Usage:
    .venv/bin/python scripts/push_ouro_sft_to_hub.py \
        --checkpoint outputs/ouro_terminal_sft/checkpoint/step-1000 \
        --repo_id jaslee/Ouro-1.4B-Thinking-terminal-sft
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi

# Files that make a custom-architecture checkpoint loadable. The weights come
# from the training run; everything else is carried over from the base model
# unless the run already emitted its own copy.
_COMPANION_FILES = [
    "configuration_ouro.py",
    "modeling_ouro.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint", required=True,
        help="Local HF-format checkpoint dir (the step-N folder torchtitan exported).",
    )
    ap.add_argument(
        "--base_model", default="./assets/hf/Ouro-1.4B-Thinking",
        help="Checkpoint the run initialised from; supplies the companion files.",
    )
    ap.add_argument("--repo_id", required=True, help="Target Hub repo, e.g. user/name.")
    ap.add_argument(
        "--private", action="store_true",
        help="Create the repo private. Off by default so the repo is shareable.",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Stage companion files and report what would upload, without uploading.",
    )
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    base = Path(args.base_model)
    if not ckpt.is_dir():
        raise SystemExit(f"checkpoint dir not found: {ckpt}")
    weights = sorted(ckpt.glob("*.safetensors"))
    if not weights:
        raise SystemExit(
            f"no *.safetensors in {ckpt} -- was the run configured with "
            "checkpoint.last_save_in_hf=True, and did it reach the last step?"
        )

    for name in _COMPANION_FILES:
        src, dst = base / name, ckpt / name
        if dst.exists():
            continue  # the run exported its own; never clobber it
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  staged {name}")

    # auto_map is what tells `trust_remote_code` loaders which classes to import
    # from the bundled modeling file. Without it a fresh clone of this repo
    # cannot be instantiated even with the .py files present.
    cfg_path = ckpt / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if "auto_map" not in cfg and (base / "config.json").exists():
            base_cfg = json.loads((base / "config.json").read_text())
            if "auto_map" in base_cfg:
                cfg["auto_map"] = base_cfg["auto_map"]
                cfg_path.write_text(json.dumps(cfg, indent=2))
                print("  restored auto_map into config.json")

    payload = sorted(p.name for p in ckpt.iterdir() if p.is_file())
    total_mb = sum(p.stat().st_size for p in ckpt.iterdir() if p.is_file()) / 1e6
    print(f"\n{len(payload)} files, {total_mb:.0f} MB -> {args.repo_id}")
    for name in payload:
        print(f"  {name}")

    if args.dry_run:
        print("\n[dry run] nothing uploaded.")
        return

    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Ouro-1.4B-Thinking terminal-agent SFT",
    )
    print(f"\nuploaded: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
