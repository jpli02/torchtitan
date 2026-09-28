"""Which terminal-bench tasks actually leaked into the SFT mix?

check_canary.py found 8 rows carrying the canary. The question that decides
whether our 4/80 is inflated is WHICH tasks those rows are about -- and in
particular whether any of the 6 tasks a checkpoint has ever solved is among
them.
"""
import os
import re
import sys

sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
os.environ.setdefault("OURO_SFT_MIX", "longhorizon")
os.environ["OURO_SFT_MIN_CMDS"] = "0"

from torchtitan.hf_datasets.text_datasets import _load_terminal_agent_sft_dataset

CANARY_PHRASE = "terminal-bench-canary"
TASKS = "/home/jli199/terminal_bench_eval/tb_tasks"
N_ROWS = int(os.environ.get("CANARY_ROWS", "4000"))
SOLVED = {"fibonacci-server", "fix-pandas-version", "fix-permissions",
          "hello-world", "heterogeneous-dates", "vim-terminal-task"}

names = sorted(d for d in os.listdir(TASKS)
               if os.path.isdir(os.path.join(TASKS, d)))

ds = _load_terminal_agent_sft_dataset("unused")
found = []
scanned = 0
for row in ds:
    msgs = row.get("messages") or []
    if not msgs:
        continue
    blob = "\n".join(str(m.get("content") or "") for m in msgs)
    scanned += 1
    if CANARY_PHRASE in blob:
        # which of our 80 task names are named in this row?
        named = [n for n in names if n in blob]
        i = blob.find(CANARY_PHRASE)
        ctx = " ".join(blob[max(0, i - 220):i + 90].split())
        found.append((scanned, named, ctx))
    if scanned >= N_ROWS:
        break

print(f"scanned {scanned} rows; {len(found)} carry the canary\n")
allnamed = set()
for idx, named, ctx in found:
    allnamed |= set(named)
    print(f"--- row {idx} | tasks named: {named or 'none'}")
    print(f"    ...{ctx[:230]}\n")

print(f"distinct tb tasks named in canary rows: {len(allnamed)}")
print(f"  {sorted(allnamed)}")
overlap = sorted(allnamed & SOLVED)
print(f"\noverlap with the 6 tasks ever solved: {overlap if overlap else 'NONE'}")
