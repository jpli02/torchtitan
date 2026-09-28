"""Is the batching decision actually supervised?

P(batch) did not move after 5k steps on batching-dense data. Either the data
never carried the signal, or the loss mask never let it through. Find, in real
tokenized+masked training rows, the positions right after a command object
closes, and report what the LABEL is there and whether it is masked out.
"""
import os, sys
os.environ.setdefault("OURO_SFT_MIX", "longhorizon")
os.environ.setdefault("OURO_SFT_MIN_CMDS", "1.5")
sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import (
    _load_terminal_agent_sft_dataset, _terminal_agent_sft_tokens,
)

tok = HuggingFaceTokenizer(tokenizer_path=
    "/home/jli199/torchtitan/assets/hf/Ouro-1.4B-Thinking")

ds = _load_terminal_agent_sft_dataset("unused")
rows = 0
dec_sup = dec_masked = 0
targets = {}
for row in ds:
    msgs = row.get("messages") or []
    if not msgs:
        continue
    ids, labels = _terminal_agent_sft_tokens(row, tok)
    if not ids:
        continue
    rows += 1
    txt = [tok.decode([i]) for i in ids]
    for k in range(len(ids) - 1):
        # a command object just closed: token containing '}' but not '}]'
        t = txt[k]
        if "}" in t and "]" not in t and '"duration"' not in t:
            nxt = txt[k + 1]
            lab = labels[k] if k < len(labels) else -100
            if lab == -100:
                dec_masked += 1
            else:
                dec_sup += 1
                key = nxt.strip()[:6] or "<ws>"
                targets[key] = targets.get(key, 0) + 1
    if rows >= 25:
        break

print(f"rows examined: {rows}")
print(f"decision points SUPERVISED: {dec_sup}   MASKED OUT: {dec_masked}")
tot = sum(targets.values()) or 1
print("what is supervised as the next token at those points:")
for k, v in sorted(targets.items(), key=lambda x: -x[1])[:8]:
    print(f"   {v/tot:8.3f}  {k!r}")
