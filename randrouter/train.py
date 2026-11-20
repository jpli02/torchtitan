"""Random-router training of Ouro-2.6B-Thinking, from the released base model (user rule 2026-11-16: every training starts there).

Text: the base model's own full-depth greedy answers to LiveCodeBench problems that are NOT in the 342-problem evaluation set
(randrouter/gen_teacher.py). Every step teacher-forces a batch of them under a router drawn at random,
  * iid   (share --p_iid)  the random router: every answer token draws its exit loop independently from a per-sequence
                           distribution over the 4 loops, itself drawn from Dirichlet(--dir_alpha);
  * rank  (share --p_rank) the same random exit shares, but the tokens that leave after a loop are the ones the released gate
                           is surest about (what a threshold router does, at every mixture of exit loops);
  * gate  (share --p_gate) the gate with random per-loop thresholds, uniform in the router searches' box [0.05, 1]^3;
  * full  (the rest)       every token at full depth (keeps the full-depth model where it is),
with the decoder's cache fill (core.routed_forward), and minimises the cross-entropy between the full-depth token and the
readout of every loop the token ran (--loss_on valid) or of its exit loop only (--loss_on exit).
Trained: LoRA updates (rank --r) of the attention and feed-forward projections of all 48 layers; the base weights, the head and
the exit gate stay frozen (the gate is refitted afterwards: randrouter/stage2.py). No reinforcement learning anywhere: the
router's choices are random draws, never optimised against a reward.

  CUDA_VISIBLE_DEVICES=0 python randrouter/train.py --tag r1 [--steps 600] [--lr 1e-4] [--budget_tok 2000] [--val_every 100]
Writes <OUT>/<tag>/{train.jsonl (one line per logged step), val.jsonl (teacher-forced fidelity of fixed routers on held-out
problems), lora_step<N>.pt, lora_last.pt, args.json}.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
import tf_eval  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="/home/jli199/boptim_scratch/randrouter")
ap.add_argument("--teacher", default="/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--min_lr_factor", type=float, default=0.1)
ap.add_argument("--budget_tok", type=int, default=2000, help="padded tokens per batch")
ap.add_argument("--max_batch", type=int, default=8)
ap.add_argument("--max_len", type=int, default=3072)
ap.add_argument("--r", type=int, default=32)
ap.add_argument("--alpha", type=float, default=64.0)
ap.add_argument("--p_iid", type=float, default=0.4)
ap.add_argument("--p_rank", type=float, default=0.4)
ap.add_argument("--p_gate", type=float, default=0.1)
ap.add_argument("--dir_alpha", type=float, default=1.0)
ap.add_argument("--loss_on", choices=["valid", "exit"], default="valid")
ap.add_argument("--loop_w", default="0.5,1,1,1", help="per-loop weights of the valid-readout loss (loop 1 never reads a filled cache slot)")
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--accum", type=int, default=1, help="batches per optimizer step (gradient accumulation)")
ap.add_argument("--reload", action="store_true", help="re-read the teacher files at every epoch (the generation is still running)")
ap.add_argument("--drop_longer", type=int, default=0, help="leave out training sequences longer than this many tokens (small card)")
ap.add_argument("--val_every", type=int, default=100)
ap.add_argument("--val_n", type=int, default=48)
ap.add_argument("--val_routers", default="full;0.3;0.245,0.3,0.3;0.2;0.25,1.0,1.0;rank:0.5,0.5,0.5;iid:0.25,0.25,0.25,0.25")
ap.add_argument("--train_n", type=int, default=10**9, help="use only the first N training sequences (data-size ablation)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--train_split", default="train")
ap.add_argument("--val_split", default="val")
ap.add_argument("--mem_gb", type=float, default=0.0)
ap.add_argument("--no_ckpt", action="store_true", help="no gradient checkpointing (faster, much more memory)")
args = ap.parse_args()

run_dir = os.path.join(args.out, args.tag)
os.makedirs(run_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(run_dir, "args.json"), "w"), indent=1)
if args.mem_gb:
    torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
torch.manual_seed(args.seed)
gen = torch.Generator().manual_seed(args.seed)

model, tok = core.load_model(core.SERVE_DIR)            # the released base weights
assert os.path.realpath(os.path.join(core.SERVE_DIR, "model.safetensors")) == os.path.realpath(os.path.join(core.BASE_DIR, "model.safetensors")), \
    "the serving directory does not hold the released base weights"
params = core.add_lora(model, args.r, args.alpha)
n_train = sum(p.numel() for p in params)
opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    import math
    prog = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (args.min_lr_factor + (1 - args.min_lr_factor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


train = core.load_teacher(args.teacher, split=args.train_split, max_len=args.max_len)[:args.train_n]
val = core.load_teacher(args.teacher, split=args.val_split, max_len=args.max_len)[:args.val_n]
val_routers = [tf_eval.parse_router(s) for s in args.val_routers.split(";")]
print(f"trainable LoRA parameters {n_train / 1e6:.1f} M; train {len(train)} sequences ({sum(len(s['ids']) - s['n_prompt'] for s in train)} answer tokens), "
      f"val {len(val)}; run dir {run_dir}", flush=True)


def validate(step):
    model.eval()
    t0 = time.time()
    res = tf_eval.evaluate(model, val, val_routers, tok.pad_token_id, budget_tok=max(args.budget_tok, 3072), log=lambda s: print(s, flush=True))
    with open(os.path.join(run_dir, "val.jsonl"), "a") as f:
        f.write(json.dumps({"step": step, "routers": res}) + "\n")
    torch.save(core.lora_state(model), os.path.join(run_dir, "lora_last.pt"))
    torch.save(core.lora_state(model), os.path.join(run_dir, f"lora_step{step}.pt"))
    print(f"[val step {step}] {time.time() - t0:.0f}s", flush=True)
    model.train()


validate(0)
step, epoch = 0, 0
t_start = time.time()
agg = {}
model.train()
micro = 0
LOOP_W = [float(v) for v in args.loop_w.split(",")]
assert len(LOOP_W) == core.N_LOOPS
if args.drop_longer:
    train = [s_ for s_ in train if len(s_["ids"]) <= args.drop_longer]
    print(f"training sequences of at most {args.drop_longer} tokens: {len(train)}", flush=True)
assert train, "no training sequence yet"
opt.zero_grad(set_to_none=True)
while step < args.steps:
    if args.reload and epoch > 0:
        train = core.load_teacher(args.teacher, split=args.train_split, max_len=args.max_len)[:args.train_n]
        if args.drop_longer:
            train = [s_ for s_ in train if len(s_["ids"]) <= args.drop_longer]
        print(f"epoch {epoch + 1}: {len(train)} training sequences", flush=True)
    batches = core.make_batches(train, args.budget_tok, args.max_batch, shuffle_gen=gen)
    epoch += 1
    for batch in batches:
        if step >= args.steps:
            break
        ids, n_prompt, lengths = core.collate(train, batch, tok.pad_token_id)
        u = float(torch.rand((), generator=gen))
        plan = thr = shares = None
        if u < args.p_iid:
            kind = "iid"
            plan, _ = core.sample_iid_plan(ids, gen, alpha=args.dir_alpha)
        elif u < args.p_iid + args.p_rank:
            kind = "rank"
            shares, _ = core.sample_shares(ids.shape[0], gen, alpha=args.dir_alpha)
        elif u < args.p_iid + args.p_rank + args.p_gate:
            kind = "gate"
            thr = core.sample_thresholds(gen)
        else:
            kind = "full"
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = core.routed_forward(model, ids, n_prompt, plan=plan, thresholds=thr, shares=shares, lengths=lengths, grad_ckpt=not args.no_ckpt)
            loss, st = core.readout_stats(model, out, ids, n_prompt, lengths, loss_on=args.loss_on, loop_w=LOOP_W)
        (loss / args.accum).backward()
        micro += 1
        for k, v in st.items():
            agg[k] = agg.get(k, 0.0) + v
        agg["loss"] = agg.get("loss", 0.0) + float(loss.detach())
        agg["micro"] = agg.get("micro", 0) + 1
        agg[f"kind_{kind}"] = agg.get(f"kind_{kind}", 0) + 1
        del out, loss
        if micro % args.accum:
            continue
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        agg["gnorm"] = agg.get("gnorm", 0.0) + gnorm
        agg["steps"] = agg.get("steps", 0) + 1
        if step % 10 == 0 or step == args.steps:
            n = agg["steps"]
            rec = {"step": step, "epoch": epoch, "loss": agg["loss"] / agg["micro"], "gnorm": agg["gnorm"] / n, "lr": lr_at(step - 1),
                   "loops": agg["loops_sum"] / max(agg["n_read"], 1), "exit_agree": agg["exit_ok"] / max(agg["n_read"], 1),
                   "exit_ce": agg["exit_ce"] / max(agg["n_read"], 1),
                   "ce_by_loop": [agg[f"ce{t}"] / max(agg[f"n{t}"], 1) for t in range(4)],
                   "agree_by_loop": [agg[f"ok{t}"] / max(agg[f"n{t}"], 1) for t in range(4)],
                   "kinds": {k[5:]: v for k, v in agg.items() if k.startswith("kind_")},
                   "s_per_step": (time.time() - t_start) / step, "mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            with open(os.path.join(run_dir, "train.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"step {step} ep {epoch} loss {rec['loss']:.4f} exit-agree {100 * rec['exit_agree']:.2f}% loops {rec['loops']:.2f} "
                  f"ce/loop {[round(c, 3) for c in rec['ce_by_loop']]} gnorm {rec['gnorm']:.2f} lr {rec['lr']:.2e} "
                  f"{rec['s_per_step']:.1f}s/step mem {rec['mem_gb']:.1f}G", flush=True)
            agg = {}
        if step % args.val_every == 0 or step == args.steps:
            validate(step)
print(f"TRAIN DONE {args.tag} steps {step} {time.time() - t_start:.0f}s", flush=True)
