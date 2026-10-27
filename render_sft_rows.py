"""Render SFT rows with the training renderer and dump token ids + labels.

Usage: python -u render_sft_rows.py V3_TRAIN.jsonl MIXED.jsonl OUT.jsonl STEPS
Runs in the torchtitan venv (needs the fork's renderer). Writes one JSON object
per row: {"i", "group" ("new"|"replay"), "ids", "labels"} and prints the token
budget: total / supervised tokens per group and the TRUE epoch count
(= steps * 4096 / total tokens, because training packs rows into 4096-token
windows, so steps != rows).
"""
import hashlib
import json
import os
import statistics
import sys
import time

os.environ.setdefault("OURO_SFT_MIX", "tb80sft")
sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")

from torchtitan.components.tokenizer import HuggingFaceTokenizer  # noqa: E402
from torchtitan.hf_datasets.text_datasets import _terminal_agent_sft_tokens  # noqa: E402

SEQ = 4096
v3file, mixed, out, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])


def key(row):
    return hashlib.md5(json.dumps(row["messages"], sort_keys=True).encode()).hexdigest()


v3 = {key(json.loads(l)) for l in open(v3file) if l.strip()}
tok = HuggingFaceTokenizer(tokenizer_path="/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking")
stats = {"new": [], "replay": []}
t0 = time.time()
with open(out, "w") as fo:
    for i, line in enumerate(open(mixed)):
        if not line.strip():
            continue
        row = json.loads(line)
        ids, labels = _terminal_agent_sft_tokens(row, tok)
        grp = "replay" if key(row) in v3 else "new"
        fo.write(json.dumps({"i": i, "group": grp, "ids": ids, "labels": labels}) + "\n")
        stats[grp].append((len(ids), sum(1 for l in labels if l != -100), len(row["messages"])))
        if (i + 1) % 50 == 0:
            print(f"  {i+1} rows, {time.time()-t0:.0f}s", flush=True)

total = sum(n for g in stats.values() for n, _, _ in g)
sup = sum(s for g in stats.values() for _, s, _ in g)
print(f"\n{os.path.basename(mixed)}: rows={sum(len(g) for g in stats.values())} total_tokens={total:,} "
      f"supervised={sup:,} ({sup/total:.0%}) steps={steps} -> tokens seen={steps*SEQ:,} = {steps*SEQ/total:.2f} epochs")
for g, rows in stats.items():
    if not rows:
        continue
    n = [r[0] for r in rows]
    s = [r[1] for r in rows]
    m = [r[2] for r in rows]
    print(f"  {g:6s}: rows={len(rows):3d} tokens={sum(n):,} ({sum(n)/total:.0%}) mean/row={statistics.mean(n):,.0f} "
          f"median={statistics.median(n):,.0f} max={max(n):,} rows>{SEQ}tok={sum(x > SEQ for x in n)} "
          f"supervised/row={statistics.mean(s):,.0f} msgs/row={statistics.mean(m):.1f}")
print(f"wrote {out} in {time.time()-t0:.0f}s")
