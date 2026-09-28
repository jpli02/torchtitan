#!/usr/bin/env python3
"""Mine TerminalTraj (the bulk source) for trajectories that exercise a target
skill, to thicken skills the tb80sft build left thin (fasttext/jupyter/tmux/
qemu -- 8/11/13/92 trajectories, vs the tools' benchmark tasks needing them).

Unlike build_bulk(), which reweights by *domain* and caps to a token target,
this keeps EVERY complete trajectory whose assistant `keystrokes` actually run
one of the requested tools, so a rare tool is no longer thinned out by the
domain sampler. Rows come out in the same {source, domain, skill, messages}
shape build_tb80_sft writes, already terminus-2 (TerminalTraj is native
terminus-2 JSON), so they append directly to the bulk pool.

Cannot come from generation: Nemotron synthetic tasks have no qemu/tmux/jupyter
/fasttext category, and the paid teacher APIs are out of credit -- so mining
existing verified corpora is the available lever.

  python skill_mine.py --skills fasttext jupyter tmux qemu \
      --max_scan 400000 --per_skill 400 --out /home/jli199/boptim_scratch/tb80sft/skill_thicken.jsonl
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
from datasets import load_dataset  # noqa: E402

from measure_corpus import classify, task_text  # noqa: E402
from torchtitan.hf_datasets.text_datasets import (  # noqa: E402
    _ends_complete,
    _normalise_terminal_messages,
)

# reuse the audit's tool signatures
from audit_skill_coverage import SKILL_PAT  # noqa: E402

TB = "/home/jli199/terminal_bench_eval/tb_tasks"
OURS = set(d for d in os.listdir(TB) if os.path.isdir(os.path.join(TB, d)))
# eval-task title fragments to keep contamination out of the mined bulk
OUR_FRAGS = [t.replace("-", " ") for t in OURS] + [t.replace("-", "") for t in OURS]


def keystrokes(msgs):
    out = []
    for m in msgs:
        if m["role"] != "assistant":
            continue
        for ks in re.findall(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"', m["content"]):
            out.append(ks)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", nargs="+", required=True)
    ap.add_argument("--dataset", default="m-a-p/TerminalTraj")
    ap.add_argument("--max_scan", type=int, default=400000)
    ap.add_argument("--per_skill", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pats = {s: re.compile(SKILL_PAT[s], re.I) for s in a.skills}
    kept = collections.Counter()
    scanned = complete = 0
    ds = load_dataset(a.dataset, split="train", streaming=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in ds:
            scanned += 1
            if scanned > a.max_scan:
                break
            if scanned % 20000 == 0:
                print(f"  scanned {scanned}, kept {dict(kept)}", flush=True)
            if all(kept[s] >= a.per_skill for s in a.skills):
                break
            msgs = _normalise_terminal_messages(r.get("messages"))
            if not msgs or not _ends_complete(msgs):
                continue
            complete += 1
            tt = task_text(msgs)
            low = tt.lower()
            if any(fr in low for fr in OUR_FRAGS if len(fr) > 10):
                continue  # looks like one of the 80 eval tasks
            ks = keystrokes(msgs)
            for s, pat in pats.items():
                if kept[s] >= a.per_skill:
                    continue
                if pat.search(ks):
                    f.write(json.dumps({"source": "terminaltraj-skill", "skill": s,
                                        "domain": classify(tt), "messages": msgs}) + "\n")
                    kept[s] += 1
                    break
    print(f"scanned {scanned} rows ({complete} complete); kept per skill: {dict(kept)}")
    print(f"-> {a.out}  ({sum(kept.values())} rows)")


if __name__ == "__main__":
    main()
