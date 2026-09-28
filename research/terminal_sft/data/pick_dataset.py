"""Which single dataset is closest to terminal-bench, so we can SFT on just it?

Every mix so far blended 3-4 corpora, which makes any result hard to attribute.
Score each corpus alone on the three axes that plausibly decide agent success:

  format   fraction of assistant turns that are valid terminus-2 JSON with a
           commands LIST. terminal-bench's terminus-2 agent parses exactly this;
           a corpus in another format teaches the wrong output shape.
  horizon  assistant turns per trajectory. terminal-bench episodes run a median
           of 22, so a corpus of 6-turn sessions cannot teach the long game.
  finish   fraction ending in task_complete=true, i.e. NOT abandoned. We
           measured abandoned trajectories to be half the length of finished
           ones, so this and horizon are entangled.
"""
import json
import os
import re
import statistics
import sys

sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")

from torchtitan.hf_datasets.text_datasets import _normalise_terminal_messages

from datasets import load_dataset

N = int(os.environ.get("PICK_ROWS", "250"))
SPECS = [
    ("nvidia/Nemotron-Terminal-Corpus", "skill_based_medium", "conversations"),
    ("nvidia/Nemotron-Terminal-Corpus", "skill_based_easy", "conversations"),
    ("m-a-p/TerminalTraj", None, "messages"),
    ("open-thoughts/OpenThoughts-Agent-v1-SFT", None, "conversations"),
]


def obj(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


print(f"{'dataset':<52} {'rows':>5} {'fmt%':>6} {'turns':>12} {'finish%':>8}")
print("-" * 90)
for repo, cfg, key in SPECS:
    try:
        ds = load_dataset(repo, cfg, split="train", streaming=True)
    except Exception as e:
        print(f"{repo+'/'+str(cfg):<52} LOAD FAILED: {str(e)[:28]}")
        continue
    turns, fmt_ok, fmt_tot, fin, rows = [], 0, 0, 0, 0
    for row in ds:
        raw = row.get(key)
        msgs = _normalise_terminal_messages(raw) if raw else None
        if not msgs:
            continue
        a = [m for m in msgs if m.get("role") == "assistant"]
        if not a:
            continue
        rows += 1
        turns.append(len(a))
        objs = []
        for m in a:
            fmt_tot += 1
            o = obj(str(m.get("content") or ""))
            if o is not None and isinstance(o.get("commands"), list):
                fmt_ok += 1
                objs.append(o)
        if objs and objs[-1].get("task_complete") is True:
            fin += 1
        if rows >= N:
            break
    if not rows:
        print(f"{repo+'/'+str(cfg):<52} no usable rows")
        continue
    label = f"{repo}{'/'+cfg if cfg else ''}"
    print(f"{label:<52} {rows:>5} {100*fmt_ok/max(fmt_tot,1):>5.0f}% "
          f"{statistics.mean(turns):>6.1f}/{statistics.median(turns):<5.0f} "
          f"{100*fin/rows:>7.0f}%")
print("\nterminal-bench reference: median 22 turns, terminus-2 JSON format")
