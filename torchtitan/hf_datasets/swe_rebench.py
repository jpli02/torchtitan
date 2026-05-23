# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face dataset `nebius/SWE-rebench-openhands-trajectories` for coding SFT.

Each row contains a multi-turn `trajectory` (OpenHands-style messages). Use with
:class:`~torchtitan.hf_datasets.text_datasets.ChatDataLoader` and
``multi_turn=True``. Trajectories are often much longer than typical SFT
contexts; raise ``training.seq_len`` (and model RoPE limits) or expect many
dropped examples.
"""

from __future__ import annotations

import json
from typing import Any

ROLE_TO_FIELD_NAMES: dict[str, list[str]] = {
    "system": ["role", "content"],
    "assistant": ["role", "content", "tool_calls"],
    "user": ["role", "content"],
    "tool": ["role", "content", "name", "tool_call_id"],
}


def filter_and_deserialize(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields needed per role and parse tool call arguments from JSON strings."""
    trajectory: list[dict[str, Any]] = []
    for msg in row["trajectory"]:
        role = msg["role"]
        if role not in ROLE_TO_FIELD_NAMES:
            raise ValueError(f"Unsupported message role in trajectory: {role!r}")
        slim: dict[str, Any] = {
            field_name: msg[field_name] for field_name in ROLE_TO_FIELD_NAMES[role]
        }
        if slim["role"] == "assistant" and slim.get("tool_calls") is not None:
            for i, tool_call in enumerate(slim["tool_calls"]):
                fn = tool_call.get("function") or {}
                raw_args = fn.get("arguments")
                if isinstance(raw_args, str):
                    slim["tool_calls"][i]["function"]["arguments"] = json.loads(raw_args)
        trajectory.append(slim)
    return {**row, "trajectory": trajectory}


def process_swe_rebench_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """``ChatDataLoader`` sample_processor: HF row -> chat messages for the template."""
    return filter_and_deserialize(dict(sample))["trajectory"]
