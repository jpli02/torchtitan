# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
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
from torchtitan.components.loss import IGNORE_INDEX
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


# Roles whose tokens should be trained on during SFT.
_SFT_TRAINABLE_ROLES = {"assistant"}


def _format_msg_for_template(msg: dict[str, Any]) -> dict[str, str]:
    """Flatten a rich trajectory message to {role, content} for the chat template.

    Complex fields (tool_calls, tool_call_id, name) are JSON-serialised and
    appended to the content string so no information is silently dropped.
    """
    role = msg["role"]
    parts: list[str] = []
    if msg.get("content"):
        parts.append(str(msg["content"]))
    if msg.get("tool_calls"):
        parts.append(json.dumps(msg["tool_calls"], ensure_ascii=False))
    if msg.get("name"):
        parts.append(f"[tool: {msg['name']}]")
    if msg.get("tool_call_id"):
        parts.append(f"[tool_call_id: {msg['tool_call_id']}]")
    return {"role": role, "content": "\n".join(parts)}


def _sft_tokens_from_messages(
    messages: list[dict[str, str]], tokenizer: BaseTokenizer
) -> tuple[list[int], list[int]]:
    """SFT tokenisation for a {role, content} message list: only assistant turns
    carry loss signal.

    Applies the tokenizer's chat template incrementally (one message at a time)
    to locate each message's exact token span.  Non-assistant tokens are replaced
    by IGNORE_INDEX in the returned label sequence.

    Returns:
        token_ids  – full sequence of token IDs
        label_ids  – parallel sequence; IGNORE_INDEX where the token is not a
                     training target, otherwise identical to token_ids
    """
    token_ids: list[int] = []
    label_ids: list[int] = []

    for i, msg in enumerate(messages):
        # Tokenize the conversation up through message i.  The boundary for
        # message i is found by diffing against the tokenization of messages[:i].
        # This requires the chat template to be prefix-consistent (the first
        # len(template(msgs[:i])) tokens of template(msgs[:i+1]) must equal
        # template(msgs[:i])).  Standard Jinja chat templates satisfy this.
        curr_tokens = tokenizer.encode(
            tokenizer.apply_chat_template(messages[: i + 1]),
            add_bos=True,
            add_eos=False,
        )
        if i > 0:
            prev_len = len(
                tokenizer.encode(
                    tokenizer.apply_chat_template(messages[:i]),
                    add_bos=True,
                    add_eos=False,
                )
            )
        else:
            prev_len = 0
        msg_tokens = curr_tokens[prev_len:]
        if msg["role"] in _SFT_TRAINABLE_ROLES:
            label_ids.extend(msg_tokens)
        else:
            label_ids.extend([IGNORE_INDEX] * len(msg_tokens))
        token_ids.extend(msg_tokens)

    # EOS always included in the training signal.
    if tokenizer.eos_id is not None:
        token_ids.append(tokenizer.eos_id)
        label_ids.append(tokenizer.eos_id)

    return token_ids, label_ids


def _swe_rebench_sft_tokens(
    sample: dict[str, Any], tokenizer: BaseTokenizer
) -> tuple[list[int], list[int]]:
    """SFT tokens for one SWE-rebench trajectory (assistant-only loss)."""
    raw_messages = _deserialize_swe_rebench_trajectory(sample)
    messages = [_format_msg_for_template(m) for m in raw_messages]
    return _sft_tokens_from_messages(messages, tokenizer)


# ---------------------------------------------------------------------------
# nvidia/OpenCodeReasoning  (R1 reasoning traces over competitive-programming
# problems).  Each row: input (problem statement), output (reasoning+solution),
# solution, plus source/dataset/split/index metadata.
#   - split_0: `input` holds the problem statement directly.
#   - split_1: `input` is the placeholder "-"; the statement must be recovered
#     from the original BAAI/TACO or codeparrot/apps row via (dataset, split,
#     index).  Those side datasets are loaded once and cached on first use.
# SFT format: a single user turn (problem) + assistant turn (reasoning+solution),
# trained assistant-only via _sft_tokens_from_messages.
# ---------------------------------------------------------------------------

_OCR_SIDE_DATASET_PATHS = {"taco": "BAAI/TACO", "apps": "codeparrot/apps"}
# Lazily-populated cache: source name -> loaded (map-style) DatasetDict.
_OCR_SIDE_DATASETS: dict[str, Any] = {}


def _load_open_code_reasoning_dataset(dataset_path: str, split: str):
    """Stream one OpenCodeReasoning split (config name == split name).

    No shuffle by default -- every run reads the same examples in the same
    order, which is why repeat "seeds" of a run that otherwise has no other
    source of randomness (e.g. Ouro's stage2_adaptive SFT, which loads the
    gate from a pretrained checkpoint rather than a random init) were
    producing byte-identical checkpoints regardless of --debug.seed. Opt in
    to a real seeded buffered shuffle via OURO_SFT_DATA_SEED so repeat runs
    can actually see different training examples; unset (the default)
    reproduces the exact prior behavior for every existing config.
    """
    ds = load_dataset(dataset_path, name=split, split=split, streaming=True)
    data_seed = os.environ.get("OURO_SFT_DATA_SEED")
    if data_seed is not None:
        ds = ds.shuffle(seed=int(data_seed), buffer_size=10_000)
    return ds


def _ocr_question(sample: dict[str, Any]) -> str:
    """Problem statement for a row: ``input`` directly, or reconstructed from the
    original TACO/APPS row for split_1 (where ``input`` is the placeholder "-")."""
    question = sample.get("input", "")
    if question != "-":
        return question

    source = sample["dataset"]
    if source not in _OCR_SIDE_DATASET_PATHS:
        raise ValueError(
            f"OpenCodeReasoning row needs reconstruction but dataset={source!r} "
            f"is not one of {list(_OCR_SIDE_DATASET_PATHS)}"
        )
    if source not in _OCR_SIDE_DATASETS:
        logger.info(f"Loading OpenCodeReasoning side dataset {source} for question reconstruction")
        _OCR_SIDE_DATASETS[source] = load_dataset(
            _OCR_SIDE_DATASET_PATHS[source], trust_remote_code=True
        )
    return _OCR_SIDE_DATASETS[source][sample["split"]][int(sample["index"])]["question"]


def _open_code_reasoning_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """One problem as a [user, assistant] chat: assistant = reasoning+solution."""
    return [
        {"role": "user", "content": _ocr_question(sample)},
        {"role": "assistant", "content": sample["output"]},
    ]


def _process_open_code_reasoning_text(sample: dict[str, Any]) -> str:
    """Non-SFT path: compact JSON of the [user, assistant] message list."""
    messages = _open_code_reasoning_messages(sample)
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))


def _open_code_reasoning_sft_tokens(
    sample: dict[str, Any], tokenizer: BaseTokenizer
) -> tuple[list[int], list[int]]:
    """SFT tokens for one OpenCodeReasoning row (assistant-only loss)."""
    return _sft_tokens_from_messages(
        _open_code_reasoning_messages(sample), tokenizer
    )


# ---------------------------------------------------------------------------
# Terminal-agent SFT mixture.
#
# Three public corpora of terminus-style agent trajectories, interleaved into a
# single stream. All three were generated by driving a strong teacher model
# through the SAME terminus-2 harness that Terminal-Bench itself uses, so their
# assistant turns are already in the exact response format the benchmark's
# parser expects (a JSON object with analysis/plan/commands/task_complete).
# That format match is the whole point of this mixture: the base Ouro model
# fails Terminal-Bench largely by emitting prose or malformed JSON that the
# harness cannot turn into actions, not by lacking shell knowledge.
#
# Sources and why each is here:
#   nvidia/Nemotron-Terminal-Corpus  -- by far the largest (~139k usable rows)
#       and the only one with published Terminal-Bench deltas at several model
#       scales; carries the bulk of the mixture weight.
#   m-a-p/TerminalTraj               -- 20k verified LONG-horizon trajectories
#       (up to 376 turns/row vs. tens elsewhere), the only source that teaches
#       staying coherent deep into a session, which is where the base model's
#       agent loop tends to fall apart.
#   open-thoughts/OpenThoughts-Agent-v1-SFT -- 15.2k rows over a narrower
#       domain (nl2bash + InferredBugs); a small auxiliary share for diversity.
#
# Deliberately NOT included:
#   allenai/TMax-SFT-16.5K / camel-ai/seta-env -- despite "SFT" in the former's
#       name, both are TASK/ENVIRONMENT definitions (task_id, description,
#       pytest validators, container defs), not chat trajectories. Using them
#       would require first generating teacher rollouts through their Docker
#       environments -- a separate pipeline stage, not a data-mixing change.
#   Nemotron's `dataset_adapters` config (226k rows) -- fails to load with
#       ArrowNotImplementedError ("Nested data conversions not implemented for
#       chunked array outputs"); the three skill_based_* configs supply far
#       more data than a 1k-step run consumes, so it is simply left out.
#
# Weights favour LONG trajectories. Measured turn counts (120-row samples):
#     Nemotron skill_based_medium   mean 11  median 12  max   18
#     Nemotron skill_based_easy     (same family, short)
#     TerminalTraj                  mean 31  median 26  max  150
#     OpenThoughts-Agent            mean 12  median 10  max   36
# The first mix put 65% of its mass on Nemotron, whose episodes never exceed 18
# turns, giving a weighted mean of ~16 turns. Terminal-Bench tasks that this
# model fails run 30-50+ turns, and the observed failure was exactly "makes
# real partial progress, then never closes out" -- i.e. it learned to act for
# about a dozen turns and declare completion. Training loss and held-out CE
# both improved monotonically (CE -64% at 1k, -70% at 10k) while task success
# did not move at all, which is what a horizon mismatch looks like: the model
# fits the data it was given, and that data is too short.
#
# TerminalTraj is the only genuinely long-horizon source available, so it now
# carries the majority of the mass. Its 20k rows are ample for a 10k-step run
# (10k steps at bs1/seq4096 = 41M tokens, and these rows are large).
_TERMINAL_SFT_SOURCES: list[tuple[str, str | None, str, float]] = [
    # (repo_id, config_name, messages_column, sampling_weight)
    ("m-a-p/TerminalTraj", None, "messages", 0.60),
    ("nvidia/Nemotron-Terminal-Corpus", "skill_based_medium", "conversations", 0.30),
    ("open-thoughts/OpenThoughts-Agent-v1-SFT", None, "conversations", 0.10),
]

# Rows whose assistant turns are all empty teach nothing (every label would be
# IGNORE_INDEX) but still consume a slot in the packed sequence buffer.
_TERMINAL_SFT_MIN_ASSISTANT_CHARS = 1

# Keep at most this many leading turns per trajectory.
#
# _sft_tokens_from_messages locates each turn's token span by re-applying the
# chat template to the whole prefix, so its cost is quadratic in turn count.
# Measured on this mixture with the Ouro tokenizer: 46 turns = 0.95s, 76 turns =
# 3.85s. TerminalTraj rows run to 376 turns, which extrapolates to ~90s to
# tokenize ONE row -- and because the dataloader tokenises inline in the
# training loop, that lands as a multi-minute stall rather than a background
# cost. Truncating to a prefix is semantically harmless for SFT (a trajectory
# prefix is itself a valid trajectory) and bounds the worst case to ~2.5s.
# 64 covers the large majority of rows in the mixture untouched.
_TERMINAL_SFT_MAX_TURNS = 64


def _normalise_terminal_messages(
    raw_messages: Any,
) -> list[dict[str, str]] | None:
    """Coerce one source row's message list into [{role, content}, ...].

    Returns None for rows that cannot contribute a training signal (malformed,
    or carrying no non-empty assistant turn), so the caller can drop them.
    """
    if not isinstance(raw_messages, (list, tuple)):
        return None

    out: list[dict[str, str]] = []
    assistant_chars = 0
    for msg in raw_messages:
        if not isinstance(msg, dict):
            return None
        # TerminalTraj/OpenThoughts/Nemotron all use {"role", "content"}, but
        # some rows in the wild use ShareGPT's {"from", "value"} spelling.
        role = msg.get("role", msg.get("from"))
        content = msg.get("content", msg.get("value"))
        if role is None or content is None:
            return None
        role = str(role)
        # ShareGPT role aliases -> ChatML roles the Ouro template understands.
        role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
        if not isinstance(content, str):
            # A few sources nest structured content (e.g. tool-call blocks) as a
            # list of parts; flatten to their text so the turn is still usable.
            if isinstance(content, (list, tuple)):
                parts = [
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                ]
                content = "".join(parts)
            else:
                content = str(content)
        if role == "assistant":
            assistant_chars += len(content.strip())
        out.append({"role": role, "content": content})

    if not out or assistant_chars < _TERMINAL_SFT_MIN_ASSISTANT_CHARS:
        return None

    if len(out) > _TERMINAL_SFT_MAX_TURNS:
        # Keep the TAIL, not the head. Task completion lives at the END of a
        # trajectory ("task_complete": true in the final assistant turn), so
        # head-truncation deletes precisely the behaviour this SFT exists to
        # teach. Measured on TerminalTraj: of the rows exceeding this cap, 8 of
        # 9 had their completion cut off by head-truncation. Every sampled row
        # in all three corpora ends in a claimed completion, so the tail is the
        # highest-value span in the row.
        #
        # A leading system/user turn carries the task statement, without which
        # the retained tail is context-free, so preserve the first turn and
        # take the last (cap - 1).
        head, tail = out[:1], out[-(_TERMINAL_SFT_MAX_TURNS - 1):]
        out = head + tail
        # The spliced tail may now begin with an assistant turn whose prompt is
        # missing; drop leading assistant turns after the preserved head so the
        # conversation still alternates sensibly.
        while len(out) > 1 and out[1]["role"] == "assistant":
            out.pop(1)
        # And never end on a non-assistant turn: it would add tokens with no
        # loss signal.
        while out and out[-1]["role"] != "assistant":
            out.pop()
        if len(out) < 2:
            return None
    return out


def _load_terminal_agent_sft_dataset(dataset_path: str):
    """Interleave the terminal-agent trajectory corpora into one stream.

    Streaming (rather than a full download) keeps this usable on a shared box:
    the corpora total >8GB on disk, while a 1k-step run touches only a small
    fraction of that. `all_exhausted` would re-loop the small sources many times
    over before the large one finishes; `first_exhausted` is the honest choice
    here since we stop at a step count, not an epoch boundary, and re-looping a
    15k-row source would just repeat data inside a single run.
    """
    from datasets import interleave_datasets

    parts = []
    weights = []
    for repo_id, config_name, messages_column, weight in _TERMINAL_SFT_SOURCES:
        ds = load_dataset(
            repo_id, config_name, split="train", streaming=True
        )
        # Normalise each source's own column name to a single "messages" field
        # so one sample_to_tokens can serve the whole mixture. remove_columns
        # drops the per-source metadata (run_id, model, ...) that would
        # otherwise make the interleaved schemas incompatible.
        ds = ds.map(
            partial(_terminal_sft_row_to_messages, messages_column=messages_column),
            remove_columns=list(ds.features) if ds.features else None,
        )
        parts.append(ds)
        weights.append(weight)

    total = sum(weights)
    probabilities = [w / total for w in weights]
    logger.info(
        "terminal_agent_sft mixture: "
        + ", ".join(
            f"{repo}{'/' + cfg if cfg else ''}={p:.0%}"
            for (repo, cfg, _, _), p in zip(_TERMINAL_SFT_SOURCES, probabilities)
        )
    )
    return interleave_datasets(
        parts,
        probabilities=probabilities,
        seed=42,
        stopping_strategy="first_exhausted",
    )


def _terminal_sft_row_to_messages(
    sample: dict[str, Any], messages_column: str
) -> dict[str, Any]:
    """Map one raw row to {"messages": [...]}; [] marks a row to skip."""
    normalised = _normalise_terminal_messages(sample.get(messages_column))
    return {"messages": normalised if normalised is not None else []}


def _process_terminal_agent_sft_text(sample: dict[str, Any]) -> str:
    """Non-SFT path: compact JSON of the normalised message list."""
    return json.dumps(
        sample.get("messages") or [], ensure_ascii=False, separators=(",", ":")
    )


def _terminal_agent_sft_tokens(
    sample: dict[str, Any], tokenizer: BaseTokenizer
) -> tuple[list[int], list[int]]:
    """SFT tokens for one mixed terminal-agent row (assistant-only loss)."""
    messages = sample.get("messages") or []
    if not messages:
        # Dropped by normalisation; contribute nothing to the packed buffer.
        return [], []
    return _sft_tokens_from_messages(messages, tokenizer)


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
    # SFT variant: identical data but loss is masked to assistant turns only.
    "swe_rebench_openhands_sft": DatasetConfig(
        path="nebius/SWE-rebench-openhands-trajectories",
        loader=_load_swe_rebench_openhands_dataset,
        sample_processor=_process_swe_rebench_openhands_text,
        sample_to_tokens=_swe_rebench_sft_tokens,
    ),
    # nvidia/OpenCodeReasoning — R1 reasoning traces. split_0's `input` is
    # self-contained (~568k rows); split_1's question is reconstructed from
    # TACO/APPS (~167k rows). SFT variants train assistant-only.
    "open_code_reasoning_split_0": DatasetConfig(
        path="nvidia/OpenCodeReasoning",
        loader=partial(_load_open_code_reasoning_dataset, split="split_0"),
        sample_processor=_process_open_code_reasoning_text,
    ),
    "open_code_reasoning_split_0_sft": DatasetConfig(
        path="nvidia/OpenCodeReasoning",
        loader=partial(_load_open_code_reasoning_dataset, split="split_0"),
        sample_processor=_process_open_code_reasoning_text,
        sample_to_tokens=_open_code_reasoning_sft_tokens,
    ),
    "open_code_reasoning_split_1": DatasetConfig(
        path="nvidia/OpenCodeReasoning",
        loader=partial(_load_open_code_reasoning_dataset, split="split_1"),
        sample_processor=_process_open_code_reasoning_text,
    ),
    "open_code_reasoning_split_1_sft": DatasetConfig(
        path="nvidia/OpenCodeReasoning",
        loader=partial(_load_open_code_reasoning_dataset, split="split_1"),
        sample_processor=_process_open_code_reasoning_text,
        sample_to_tokens=_open_code_reasoning_sft_tokens,
    ),
    # Convenience alias: the self-contained split_0 SFT set.
    "open_code_reasoning_sft": DatasetConfig(
        path="nvidia/OpenCodeReasoning",
        loader=partial(_load_open_code_reasoning_dataset, split="split_0"),
        sample_processor=_process_open_code_reasoning_text,
        sample_to_tokens=_open_code_reasoning_sft_tokens,
    ),
    # Terminal-Bench-oriented agent-trajectory mixture (see
    # _TERMINAL_SFT_SOURCES above for the per-corpus rationale). `path` is
    # unused -- the loader owns the repo list -- but DatasetConfig requires it,
    # so it names the dominant source for log readability.
    "terminal_agent_sft": DatasetConfig(
        path="nvidia/Nemotron-Terminal-Corpus",
        loader=_load_terminal_agent_sft_dataset,
        sample_processor=_process_terminal_agent_sft_text,
        sample_to_tokens=_terminal_agent_sft_tokens,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable, Callable | None]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor, config.sample_to_tokens


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

        path, dataset_loader, text_processor, sample_to_tokens = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor
        # When set, bypasses the text processor and drives the SFT label-mask path.
        self._sample_to_tokens: Callable | None = sample_to_tokens

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []
        # Parallel label buffer used only in the SFT masking path.
        # Each position holds the target token ID or IGNORE_INDEX.
        self._label_buffer: list[int] = []

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
                if self._sample_to_tokens is not None:
                    # SFT masking path: processor returns (token_ids, label_ids) directly.
                    sample_tokens, sample_labels = self._sample_to_tokens(
                        sample, self._tokenizer
                    )
                    self._token_buffer.extend(sample_tokens)
                    self._label_buffer.extend(sample_labels)
                else:
                    # Standard next-token-prediction path.
                    sample_text = self._text_processor(sample)
                    sample_tokens = self._tokenizer.encode(
                        sample_text, add_bos=True, add_eos=True
                    )
                    self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    if self._sample_to_tokens is not None:
                        # SFT: label at position i is the pre-computed target for token i+1.
                        y = torch.LongTensor(self._label_buffer[:max_buffer_token_len])
                        self._label_buffer = self._label_buffer[max_buffer_token_len:]
                        label = y[1:]
                    else:
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
        self._label_buffer = state_dict.get("label_buffer", [])
        # If the token/label buffers have different lengths (e.g., checkpoint was
        # saved from a non-SFT run that never populated label_buffer), the buffers
        # are unusable together.  Clear both so the next sample starts fresh.
        if len(self._token_buffer) != len(self._label_buffer):
            logger.warning(
                f"Discarding mismatched token/label buffers on checkpoint restore "
                f"(token={len(self._token_buffer)}, label={len(self._label_buffer)}). "
                "This is expected when switching between SFT and non-SFT datasets."
            )
            self._token_buffer = []
            self._label_buffer = []

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict: dict[str, Any] = {
            "token_buffer": self._token_buffer,
            "label_buffer": self._label_buffer,
        }

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
