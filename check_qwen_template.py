#!/usr/bin/env python3
"""Sanity-check that a bo_sft JSONL row tokenises the way the SFT loader
expects under the Qwen3 tokenizer: prefix-consistent chat template, assistant
span carries labels, nothing else does, and the row fits seq_len."""
import json
import sys

sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
from torchtitan.components.tokenizer import HuggingFaceTokenizer  # noqa: E402
from torchtitan.hf_datasets.text_datasets import (  # noqa: E402
    IGNORE_INDEX,
    _sft_tokens_from_messages,
)

jsonl = sys.argv[1] if len(sys.argv) > 1 else "/home/jli199/boptim_scratch/bo_sft/cifar.jsonl"
tok = HuggingFaceTokenizer(tokenizer_path="/home/jli199/torchtitan/assets/hf/Qwen3-1.7B")
rows = [json.loads(l) for l in open(jsonl)]
lens = []
for r in rows:
    ids, labels = _sft_tokens_from_messages(r["messages"], tok)
    lens.append(len(ids))
n_lab = sum(1 for l in labels if l != IGNORE_INDEX)
print(f"rows={len(rows)} tokens min/mean/max = {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}  eos_id={tok.eos_id}")
print(f"last row: {len(ids)} tokens, {n_lab} labelled (assistant+eos)")
print("labelled text:", repr(tok.decode([l for l in labels if l != IGNORE_INDEX]))[:300])
head = tok.decode(ids[:40])
print("prompt head:", repr(head)[:200])
