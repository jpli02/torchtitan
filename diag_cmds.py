"""How many commands does the model batch per turn, vs the training data?

terminus-2 accepts a LIST of commands per response. If a turn issues 1 command,
a task needing N shell steps costs N turns x ~20s. If it batches 4, the same
task costs N/4 turns. At a hard 420s budget that is a direct multiplier on how
far the agent gets -- and unlike 'more data', it is mechanically tied to the
measured failure (65/80 agent_timeout at ~21 turns).
"""
import json, glob, re, statistics, collections

def cmds_in(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    c = d.get("commands")
    return len(c) if isinstance(c, list) else None

# --- what OUR model emits (cleanest 80-task run) ---
counts = []
for f in glob.glob("/tmp/tb_runs/e80_c10k80_s*/*/*/agent-logs/episode-*/debug.json"):
    try:
        d = json.load(open(f)); r = d.get("original_response")
        if not (isinstance(r, str) and r.strip().startswith("{")):
            continue
        j = json.loads(r)
        msg = j["choices"][0]["message"].get("content")
        txt = msg if isinstance(msg, str) else "".join(
            b.get("text", "") for b in (msg or []))
        n = cmds_in(txt or "")
        if n is not None:
            counts.append(n)
    except Exception:
        pass
if counts:
    print(f"MODEL (continue10k, 80-task run): n={len(counts)} turns")
    print(f"  mean cmds/turn={statistics.mean(counts):.2f} "
          f"median={statistics.median(counts):.0f} max={max(counts)}")
    print(f"  distribution: {dict(sorted(collections.Counter(counts).items())[:8])}")

# --- what the TRAINING DATA shows ---
import sys
sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
from torchtitan.hf_datasets.text_datasets import _load_terminal_agent_sft_dataset

ds = _load_terminal_agent_sft_dataset("unused")
tc, n = [], 0
for row in ds:
    for m in (row.get("messages") or []):
        if m.get("role") == "assistant":
            k = cmds_in(m.get("content") or "")
            if k is not None:
                tc.append(k)
    n += 1
    if n >= 40:
        break
if tc:
    print(f"\nTRAINING DATA (longhorizon mix): n={len(tc)} assistant turns")
    print(f"  mean cmds/turn={statistics.mean(tc):.2f} "
          f"median={statistics.median(tc):.0f} max={max(tc)}")
    print(f"  distribution: {dict(sorted(collections.Counter(tc).items())[:8])}")
