"""How much of each training trajectory actually survives into the model?

seq_len is 4096 and we tail-truncate (head-truncation was dropping completions).
Nobody has measured what fraction of a trajectory that leaves. If the typical
TerminalTraj episode is much longer than 4096 tokens, then:

  - the model never sees how an episode BEGINS (orientation, first recon), and
    at eval time every episode starts at the beginning
  - the "15 median turns" we selected this corpus for is not what gets trained;
    the trained horizon is however many turns fit in the tail
  - assistant-only loss makes it worse: the supervised tokens are a subset of an
    already-truncated window

Reports the token-length distribution, the over-cap fraction, and -- for
over-cap rows -- how many assistant turns actually survive tail truncation.
"""
import os
import statistics
import sys

sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
os.environ.setdefault("OURO_SFT_MIX", "terminaltraj")
os.environ.setdefault("OURO_SFT_REQUIRE_COMPLETE", "1")

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import (
    _load_terminal_agent_sft_dataset, _sft_tokens_from_messages,
)

SEQ = int(os.environ.get("SEQ_LEN", "4096"))
N = int(os.environ.get("TRUNC_ROWS", "120"))

tok = HuggingFaceTokenizer(
    tokenizer_path="/home/jli199/torchtitan/assets/hf/Ouro-1.4B-Thinking")

lens, turns_all, surviving, over = [], [], [], 0
rows = 0
for row in _load_terminal_agent_sft_dataset("unused"):
    msgs = row.get("messages") or []
    if not msgs:
        continue
    ids, labels = _sft_tokens_from_messages(msgs, tok)
    if not ids:
        continue
    rows += 1
    n = len(ids)
    lens.append(n)
    a = [m for m in msgs if m.get("role") == "assistant"]
    turns_all.append(len(a))
    if n > SEQ:
        over += 1
        # tail truncation keeps the LAST SEQ tokens: how many assistant turns
        # start inside that window?
        cut = n - SEQ
        kept = 0
        pos = 0
        for i, m in enumerate(msgs):
            sub, _ = _sft_tokens_from_messages(msgs[: i + 1], tok)
            end = len(sub)
            if m.get("role") == "assistant" and pos >= cut:
                kept += 1
            pos = end
        surviving.append(kept)
    if rows >= N:
        break

print(f"corpus: {os.environ['OURO_SFT_MIX']}  seq_len={SEQ}  rows={rows}\n")
print(f"  tokens/trajectory : mean {statistics.mean(lens):.0f}  "
      f"median {statistics.median(lens):.0f}  "
      f"p90 {sorted(lens)[int(.9*len(lens))]}  max {max(lens)}")
print(f"  OVER the {SEQ}-token cap : {over}/{rows} ({100*over/rows:.0f}%)")
if surviving:
    print(f"  for over-cap rows, assistant turns surviving tail truncation:")
    print(f"      mean {statistics.mean(surviving):.1f}  "
          f"median {statistics.median(surviving):.0f}  "
          f"min {min(surviving)}  max {max(surviving)}")
    print(f"  (vs {statistics.mean(turns_all):.1f} mean turns in the full "
          f"trajectories)")
frac = [min(1.0, SEQ / n) for n in lens]
print(f"\n  fraction of the average trajectory that fits: "
      f"{100*statistics.mean(frac):.0f}%")
