"""Data for ONE training run from the base model (user rule 2026-11-16: every training starts from base Ouro).

  train  = v3's own training file (tb80sft/train_v3.jsonl, 7,682 rows) minus 60 rows held out for validation,
           plus the 112 take-turns r1 trajectories, SHUFFLED with a fixed seed.
  Why shuffled: the trainer reads the file in order, and train_v3.jsonl is 7,324 older rows followed by the 358 skill rows
  v3 was built for; v3's 20,000 steps read 0.905 of one pass, i.e. rows 0..6911, so none of the skill rows (and 770 rows
  in all) were ever trained on.
  validation (in-training probe, torchtitan/components/val_probe.py): the test-task oracle runs, unseen round-2 runs of the
  take-turns tasks, 15 of the 60 held-out v3 rows, and a few trained-on rows as a fit reference.

  python build_frombase_mix.py          (torchtitan venv: uses the training renderer for the held-out rows)
Writes boptim_scratch/tb80sft/train_frombase_v3r1_shuf7.jsonl, frombase_heldout_v3rows.jsonl, manifest_frombase.json and
boptim_scratch/lr_debug/lrval_frombase_rendered.jsonl.
"""
import hashlib
import json
import os
import random
import sys

os.environ.setdefault("OURO_SFT_MIX", "tb80sft")
sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
D = "/home/jli199/boptim_scratch/tb80sft"
KEEP = "/home/jli199/boptim_scratch/lr_debug"
TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
SEED, N_HELD, N_HELD_PROBE = 7, 60, 15


def key(row):
    return hashlib.md5(json.dumps(row["messages"], sort_keys=True).encode()).hexdigest()


def main():
    v3 = [json.loads(line) for line in open(f"{D}/train_v3.jsonl") if line.strip()]
    mixed = [json.loads(line) for line in open(f"{TMP}/tvary_r1_mixed.jsonl") if line.strip()]
    v3keys = {key(r) for r in v3}
    new = [r for r in mixed if key(r) not in v3keys]
    rng = random.Random(SEED)
    # held-out rows: not the extremely long ones (a 100k-token row would dominate the validation loss and its time)
    tok_counts = json.load(open(f"{KEEP}/train_v3_tokens.json"))["tokens"]
    pool = [i for i, n in enumerate(tok_counts) if n <= 20000]
    held_idx = sorted(rng.sample(pool, N_HELD))
    held = [v3[i] for i in held_idx]
    hset = set(held_idx)
    train = [r for i, r in enumerate(v3) if i not in hset] + new
    rng.shuffle(train)
    out = f"{D}/train_frombase_v3r1_shuf7.jsonl"
    with open(out, "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(f"{D}/frombase_heldout_v3rows.jsonl", "w") as f:
        for r in held:
            f.write(json.dumps(r) + "\n")
    n_skill_held = sum(1 for i in held_idx if i >= 7324)
    json.dump({"train_file": out, "rows": len(train), "v3_rows": len(v3) - N_HELD, "take_turns_r1_rows": len(new), "heldout_v3_rows": N_HELD,
               "heldout_indices_in_train_v3": held_idx, "heldout_skill_rows": n_skill_held, "shuffle_seed": SEED,
               "note": "held-out rows drawn from rows of at most 20,000 tokens; round-2 take-turns rows are NOT in training (validation)"},
              open(f"{D}/manifest_frombase.json", "w"), indent=1)
    print(f"train: {len(train)} rows = {len(v3) - N_HELD} v3 + {len(new)} take-turns r1 -> {out}")
    print(f"held out: {N_HELD} v3 rows ({n_skill_held} of them skill rows)")

    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.hf_datasets.text_datasets import _terminal_agent_sft_tokens
    tok = HuggingFaceTokenizer(tokenizer_path="/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking")
    val = []
    small = [json.loads(line) for line in open(f"{KEEP}/lrval_small_rendered.jsonl") if line.strip()]
    for g in ("oracle_test", "r2_unseen", "train_new", "train_replay"):       # train_replay rows are v3 rows that stay in training
        val += [r for r in small if r["group"] == g]
    for r in held[:N_HELD_PROBE]:
        ids, labels = _terminal_agent_sft_tokens(r, tok)
        val.append({"group": "v3_heldout", "ids": ids, "labels": labels})
    with open(f"{KEEP}/lrval_frombase_rendered.jsonl", "w") as f:
        for k, r in enumerate(val):
            f.write(json.dumps(dict(r, i=k)) + "\n")
    for g in ("oracle_test", "r2_unseen", "v3_heldout", "train_new", "train_replay"):
        rs = [r for r in val if r["group"] == g]
        print(f"  validation {g:13s} rows {len(rs):3d} tokens {sum(len(r['ids']) for r in rs):8,d} supervised {sum(sum(1 for x in r['labels'] if x != -100) for r in rs):8,d}")
    print("validation tokens in all:", sum(len(r["ids"]) for r in val))


if __name__ == "__main__":
    main()
