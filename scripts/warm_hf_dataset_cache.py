# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Materialize HuggingFace `datasets` cache once (single process).

Use before multi-GPU torchrun if your default cache is on NFS or you see
stragglers during "Generating train split" across ranks.

Example (from repo root, same HF cache as training):

    HF_DATASETS_CACHE=/tmp/torchtitan_hf_datasets_$USER python scripts/warm_hf_dataset_cache.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="tests/assets/c4_test",
        help="Path passed to load_dataset() (default: bundled c4_test)",
    )
    args = parser.parse_args()
    path = Path(args.dataset_path)
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[1]
        path = (root / path).resolve()
    load_dataset(str(path), split="train")
    cache = os.environ.get("HF_DATASETS_CACHE")
    cache_msg = cache if cache else "default (HF_HOME/datasets or ~/.cache/huggingface)"
    print(f"Warmed cache for {path} (HF_DATASETS_CACHE={cache_msg})")


if __name__ == "__main__":
    main()
