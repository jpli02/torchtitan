# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


_SWE_REBENCH_ROLE_FIELDS: dict[str, tuple[str, ...]] = {
    "system": ("role", "content"),
    "assistant": ("role", "content", "tool_calls"),
    "user": ("role", "content"),
    "tool": ("role", "content", "name", "tool_call_id"),
}


def _load_swe_rebench_openhands_dataset(dataset_path: str):
    """HF coding trajectories (OpenHands-style); streamed for scale."""
    return load_dataset(dataset_path, split="train", streaming=True)


def _deserialize_swe_rebench_trajectory(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep per-role fields and parse string tool ``function.arguments`` as JSON."""
    trajectory: list[dict[str, Any]] = []
    for raw in sample["trajectory"]:
        role = raw["role"]
        if role not in _SWE_REBENCH_ROLE_FIELDS:
            raise ValueError(f"Unknown trajectory role {role!r}")
        field_names = _SWE_REBENCH_ROLE_FIELDS[role]
        msg = {name: copy.deepcopy(raw[name]) for name in field_names}
        if msg["role"] == "assistant" and msg.get("tool_calls") is not None:
            for i, tool_call in enumerate(msg["tool_calls"]):
                fn = tool_call.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    msg["tool_calls"][i]["function"]["arguments"] = json.loads(args)
        trajectory.append(msg)
    return trajectory


def _process_swe_rebench_openhands_text(sample: dict[str, Any]) -> str:
    """One training document: compact JSON of the deserialized message list."""
    trajectory = _deserialize_swe_rebench_trajectory(sample)
    return json.dumps(trajectory, ensure_ascii=False, separators=(",", ":"))


# Add your dataset here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
    "swe_rebench_openhands": DatasetConfig(
        path="nebius/SWE-rebench-openhands-trajectories",
        loader=_load_swe_rebench_openhands_dataset,
        sample_processor=_process_swe_rebench_openhands_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict: dict[str, Any] = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


class HuggingFaceTextDataLoader(ParallelAwareDataloader):
    """Configurable text dataloader that wraps HuggingFaceTextDataset.

    This dataloader can be used for both training and validation by
    configuring the appropriate dataset, seq_len, batch_size, etc.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "c4_test"
        """Dataset to use"""

        infinite: bool = True
        """Whether to loop the dataset infinitely"""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        hf_ds = HuggingFaceTextDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
        }

        super().__init__(
            hf_ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
