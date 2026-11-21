"""Teacher data for training exit gates on BFCL (gate NAS, user 2026-11-21): the released model's own full-depth answers to BFCL v3
LIVE items (the router searches evaluate on the non-live items: search 575 / held-out 575, so no evaluated item is trained on).

  python randrouter/bfcl_teacher.py prep  [--n 600]       the n live items with the shortest prompts (they fit the 9 GB cards),
                                                          split into two shard files under <OUT>
  (then depth_eval.py ... --task bfcl --steps 4 --assistant_prefix "[" --max_new 256 --dump_ids on each shard)
  python randrouter/bfcl_teacher.py convert               shard outputs -> <OUT>/teacher_bfcl_live.jsonl (the teacher format of core.load_teacher)
"""
import argparse
import glob
import json
import os

LIVE = "/srv/scratch/jli199/bfcl_stage/bfcl_live_all.jsonl"
OUT = "/home/jli199/boptim_scratch/randrouter/bfcl_teacher"


def prep(n):
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(LIVE) if l.strip()]
    rows.sort(key=lambda r: len(r["prompt"]))
    sel = rows[:n]
    half = (len(sel) + 1) // 2
    for k, part in enumerate((sel[:half], sel[half:])):
        with open(f"{OUT}/live_short_{k}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} live items; {len(sel)} shortest (prompt up to {len(sel[-1]['prompt'])} characters) in 2 shards of {half} / {len(sel) - half}")


def convert():
    out, n_tok = [], 0
    for p in sorted(glob.glob(f"{OUT}/gen_live_short_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            if "prompt_ids" not in r or not r.get("gen_ids"):
                continue
            out.append({"question_id": r.get("id"), "category": r.get("category"), "prompt_ids": r["prompt_ids"], "gen_ids": r["gen_ids"],
                        "split": "train", "truncated": bool(r.get("truncated"))})
            n_tok += len(r["gen_ids"])
    with open(f"{OUT}/teacher_bfcl_live.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} teacher rows, {n_tok} answer tokens -> {OUT}/teacher_bfcl_live.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "convert"])
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    prep(a.n) if a.cmd == "prep" else convert()
