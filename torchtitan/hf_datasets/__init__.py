# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field


__all__ = ["DatasetConfig"]


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    sample_processor: Callable
    # Optional: if set, replaces sample_processor. Receives (sample, tokenizer) and
    # returns (token_ids, label_ids) directly, with IGNORE_INDEX already applied for
    # SFT masking. When None the text-based sample_processor path is used.
    sample_to_tokens: Callable | None = field(default=None)
