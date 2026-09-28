"""What does OURO_SFT_MIN_CMDS actually keep, and does it move the target
distribution the model is collapsing onto?

The model emits 1.0 cmds/turn; the unfiltered longhorizon mix averages 1.72.
If filtering raises the mix's mean materially AND still yields enough rows to
train on, it is a viable fix. If it starves the stream, it is not.
"""
import os, sys, statistics, collections
sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
from torchtitan.hf_datasets.text_datasets import (
    _load_terminal_agent_sft_dataset, _row_mean_commands, _MIN_CMDS,
)

ds = _load_terminal_agent_sft_dataset("unused")
kept, dropped, means, per_turn = 0, 0, [], []
it = iter(ds)
for _ in range(400):
    try:
        row = next(it)
    except StopIteration:
        break
    m = row.get("messages") or []
    if not m:
        dropped += 1
        continue
    kept += 1
    means.append(_row_mean_commands(m))

print(f"OURO_SFT_MIN_CMDS = {_MIN_CMDS}")
print(f"rows kept={kept}  dropped/empty={dropped}  "
      f"keep_rate={100*kept/max(kept+dropped,1):.0f}%")
if means:
    print(f"kept rows' mean cmds/turn: mean={statistics.mean(means):.2f} "
          f"median={statistics.median(means):.2f} max={max(means):.1f}")
    buckets = collections.Counter(min(int(x), 5) for x in means)
    print(f"distribution of row means: {dict(sorted(buckets.items()))}")
print("\nreference: unfiltered longhorizon mix = 1.72 mean; model emits 1.00")
