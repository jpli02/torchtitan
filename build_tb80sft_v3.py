#!/usr/bin/env python3
"""tb80sft v3 = v2 (kept verbatim, including its x5 on-dist upsampling) + the
skill_mine.py rows that are not already present, deduped by message content.

Thickens the four thin skills the coverage audit flagged (trajectory counts):
  qemu     92 -> 359   (also lifts kernel/boot 28 -> 55: qemu tasks build kernels)
  tmux     13 -> 45
  jupyter  11 -> 44
  fasttext  8 -> 34
Every one of the 80 tasks now has >=5 trajectories for each skill it needs.
"""
import hashlib
import json
import random

V2 = "/home/jli199/boptim_scratch/tb80sft/train.jsonl"
SKILL = "/home/jli199/boptim_scratch/tb80sft/skill_thicken.jsonl"
OUT = "/home/jli199/boptim_scratch/tb80sft/train_v3.jsonl"
MANIFEST = "/home/jli199/boptim_scratch/tb80sft/manifest_v3.json"


def key(msgs):
    return hashlib.md5(json.dumps([m["content"] for m in msgs]).encode()).hexdigest()


def main():
    random.seed(42)
    v2 = [json.loads(l) for l in open(V2)]
    present = {key(r["messages"]) for r in v2}
    added = []
    by_skill = {}
    for l in open(SKILL):
        r = json.loads(l)
        k = key(r["messages"])
        if k in present:
            continue
        present.add(k)
        added.append({"messages": r["messages"]})
        by_skill[r.get("skill", "?")] = by_skill.get(r.get("skill", "?"), 0) + 1
    out = v2 + added
    random.shuffle(out)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")
    json.dump({"v2_rows": len(v2), "skill_rows_added": len(added),
               "added_by_skill": by_skill, "total": len(out)},
              open(MANIFEST, "w"), indent=1)
    print(f"v3: {len(v2)} v2 + {len(added)} skill = {len(out)} -> {OUT}")
    print(f"added by skill: {by_skill}")


if __name__ == "__main__":
    main()
