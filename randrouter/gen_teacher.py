"""Full-depth greedy answers of the released base model to the training pool, WITH token ids (the text the random-router
training teacher-forces, and the labels of its loss).

Prompt and decoding are those of the router searches (depth_eval.py --task lcb --assistant_prefix '```python\\n', greedy,
1,024 new tokens), batched; the released weights at full depth (4 loops, no early exit).

  CUDA_VISIBLE_DEVICES=0 python randrouter/gen_teacher.py [--data FILE] [--out FILE] [--batch 6] [--budget_tok 9500] [--max_new 1024] [--splits val,train]
Appends one row per problem to --out: {"question_id", "split", "difficulty", "prompt_ids", "gen_ids", "truncated"}; problems
already in --out are skipped, so the run can be stopped and continued. Several processes can share the work: each writes its
own --out and lists the others' files in --also; a second process started with --reverse meets the first one in the middle.
"""
import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/home/jli199/boptim_scratch/ouro_base_kvshare")  # released weights (symlink) + the serving model file
ap.add_argument("--data", default="/home/jli199/boptim_scratch/randrouter/lcb_v1v4_notests.jsonl")
ap.add_argument("--out", default="/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4.jsonl")
ap.add_argument("--batch", type=int, default=6)
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--budget_tok", type=int, default=9500)
ap.add_argument("--splits", default="val,train")
ap.add_argument("--shard", default="0/1", help="i/k: this process takes every k-th batch starting at i (several processes, one file each)")
ap.add_argument("--reverse", action="store_true", help="work through the batches from the end (a second process meeting the first)")
ap.add_argument("--also", default="", help="comma-separated output files of other processes: their problems are skipped too (re-read before every batch)")
ap.add_argument("--mem_gb", type=float, default=0.0, help="cap this process's GPU memory (shared card)")
args = ap.parse_args()

APFX = "```python\n"


def build_prompt(r):
    """LiveCodeBench, official prompt shape (the same text as depth_eval.build_prompt for --task lcb)."""
    q = "### Question:\n" + r["question_content"] + "\n\n"
    if r.get("starter_code", "").strip():
        q += ("### Format: You will use the following starter code to write the solution to the problem and enclose your code "
              f"within delimiters.\n```python\n{r['starter_code']}\n```\n\n")
    else:
        q += ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the "
              "sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads "
              "the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n")
    return q + "### Answer: (use the provided format with backticks)\n\n"


rows = [json.loads(l) for l in open(args.data) if l.strip()]
done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["question_id"] for l in open(args.out) if l.strip()}
tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
# order: validation problems first, then training problems in a fixed random order (so a partial file is an unbiased sample),
# batched in groups of similar prompt length
order = []
for split in args.splits.split(","):
    part = [r for r in rows if r["split"] == split]
    random.Random(0).shuffle(part)
    order += part
texts = {r["question_id"]: tok.apply_chat_template([{"role": "user", "content": build_prompt(r)}], tokenize=False, add_generation_prompt=True) + APFX
         for r in order}
# batches of similar prompt length inside groups of 48 problems; a batch holds at most --batch problems and at most --budget_tok
# cache tokens (problems x (longest prompt + max_new)): the 4-loop cache is about 1.6 MB per token, so 9,500 tokens is 15 GB
ntok = {q: len(tok(t)["input_ids"]) for q, t in texts.items()}
GROUP = 48
batches = []
for g in range(0, len(order), GROUP):
    grp = sorted(order[g:g + GROUP], key=lambda r: ntok[r["question_id"]])
    cur = []
    for r in grp:
        if cur and (len(cur) >= args.batch or (len(cur) + 1) * (ntok[r["question_id"]] + args.max_new) > args.budget_tok):
            batches.append(cur)
            cur = []
        cur.append(r)
    if cur:
        batches.append(cur)
if args.reverse:
    batches = batches[::-1]
si, sk = (int(v) for v in args.shard.split("/"))
batches = [b for j, b in enumerate(batches) if j % sk == si]


def others_done():
    ids = set()
    for f in [x for x in args.also.split(",") if x]:
        if os.path.exists(f):
            for l in open(f):
                try:
                    ids.add(json.loads(l)["question_id"])
                except Exception:  # a line being written
                    pass
    return ids


todo = sum(1 for b in batches for r in b if r["question_id"] not in done)
print(f"{len(order)} problems, {len(batches)} batches for shard {args.shard}, {todo} problems to generate", flush=True)

if args.mem_gb:
    torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
model = AutoModelForCausalLM.from_pretrained(args.ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
assert getattr(model.model, "decode_early_exit_threshold", None) is None and model.config.total_ut_steps == 4
t0 = time.time()
n_tok = n_done = 0
with torch.inference_mode(), open(args.out, "a") as out:
    for batch in batches:
        skip = done | others_done()
        batch = [r for r in batch if r["question_id"] not in skip]
        if not batch:
            continue
        enc = tok([texts[r["question_id"]] for r in batch], return_tensors="pt", padding=True).to("cuda")
        try:
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        except torch.OutOfMemoryError:  # a long prompt on a memory-capped card: leave the batch to another process
            print(f"  out of memory on a batch of {len(batch)} (prompt {enc['input_ids'].shape[1]} tokens); skipped", flush=True)
            del enc
            torch.cuda.empty_cache()
            continue
        gen = gen[:, enc["input_ids"].shape[1]:]
        for j, r in enumerate(batch):
            p_ids = enc["input_ids"][j][enc["attention_mask"][j].bool()].tolist()
            g = gen[j].tolist()
            # the answer ends at the first end-of-sequence token (kept); what follows is batch padding
            if tok.eos_token_id in g:
                g = g[:g.index(tok.eos_token_id) + 1]
                trunc = False
            else:
                trunc = True
            out.write(json.dumps({"question_id": r["question_id"], "split": r["split"], "difficulty": r["difficulty"],
                                  "prompt_ids": p_ids, "gen_ids": g, "truncated": trunc}) + "\n")
            n_tok += len(g)
            n_done += 1
        out.flush()
        dt = time.time() - t0
        print(f"  {n_done}/{todo} problems, {dt:.0f}s, {n_tok / dt:.1f} tok/s, eta {dt / n_done * (todo - n_done) / 60:.0f} min", flush=True)
print(f"DONE {n_done} problems, {n_tok} tokens, {time.time() - t0:.0f}s", flush=True)
