"""Is the SFT data distributionally like terminal-bench, or is this a data problem?

Compares the training mix against what the benchmark actually demands on the
axes that plausibly matter for agent success:

  turns per trajectory   terminal-bench episodes run 20-35 turns before the
                         clock stops them. If the data's trajectories are much
                         shorter, the model never learns long-horizon recovery.
  ends in success        we train on assistant tokens regardless of whether the
                         trajectory SOLVED anything. Training on failed
                         trajectories teaches failure, and nothing in our
                         pipeline filters for it.
  first-turn length      a proxy for how much the data front-loads reasoning.
"""
import collections
import json
import os
import re
import statistics
import sys

sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
os.environ.setdefault("OURO_SFT_MIX", "longhorizon")
os.environ["OURO_SFT_MIN_CMDS"] = "0"

from torchtitan.hf_datasets.text_datasets import _load_terminal_agent_sft_dataset

N = int(os.environ.get("DIST_ROWS", "1500"))


def obj(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


ds = _load_terminal_agent_sft_dataset("unused")
turns, ends_complete, any_complete, first_len = [], 0, 0, []
rows = 0
for row in ds:
    msgs = row.get("messages") or []
    if not msgs:
        continue
    asst = [m for m in msgs if m.get("role") == "assistant"]
    if not asst:
        continue
    rows += 1
    turns.append(len(asst))
    first_len.append(len(str(asst[0].get("content") or "")))
    objs = [obj(str(m.get("content") or "")) for m in asst]
    objs = [o for o in objs if o]
    if objs and objs[-1].get("task_complete") is True:
        ends_complete += 1
    if any(o.get("task_complete") is True for o in objs):
        any_complete += 1
    if rows >= N:
        break

print(f"SFT mix (unfiltered longhorizon), {rows} trajectories\n")
print(f"  assistant turns : mean {statistics.mean(turns):.1f}  "
      f"median {statistics.median(turns):.0f}  "
      f"p90 {sorted(turns)[int(.9*len(turns))]}  max {max(turns)}")
b = collections.Counter(min(t // 5 * 5, 40) for t in turns)
print(f"  turn histogram  : {dict(sorted(b.items()))}  (bucketed by 5, 40=40+)")
print(f"  ends in task_complete=true : {ends_complete}/{rows} "
      f"({100*ends_complete/rows:.0f}%)")
print(f"  contains task_complete=true: {any_complete}/{rows} "
      f"({100*any_complete/rows:.0f}%)")
print(f"  first assistant turn chars : median "
      f"{statistics.median(first_len):.0f}")

print("\nwhat terminal-bench demands (measured from our own eval logs):")
import glob
ep = []
for run in ["e80_batching80_s0", "e80_batching80_s1"]:
    for d in glob.glob(f"/tmp/tb_runs/{run}/*/*/agent-logs"):
        n = len(glob.glob(f"{d}/episode-*"))
        if n:
            ep.append(n)
if ep:
    print(f"  episodes per trial : mean {statistics.mean(ep):.1f}  "
          f"median {statistics.median(ep):.0f}  max {max(ep)}  (n={len(ep)})")
    short = sum(1 for t in turns if t < statistics.median(ep))
    print(f"  SFT trajectories SHORTER than the median episode: "
          f"{short}/{rows} ({100*short/rows:.0f}%)")
