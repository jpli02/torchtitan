"""Stage II again: refit the exit gate (the router's signal) with the language model frozen, on states that come from the
DECODER'S cache under a random router, starting from a zero or random gate (never from the released gate's weights).

The paper's Stage II (arXiv 2510.25741, Eq. 6; torchtitan/components/loss.py) trains the gate on full-depth states: the gate
after loop t should say "continue" when loop t improved the loss over loop t-1 by more than gamma. Here the same language model
(the released one, or the random-router-trained one via --lora) is run under a random router with the real cache fill, and
several gates, all a linear map of the 2,048-dim state like the released one, are fitted at once on the same states:

  paper    the paper's rule on the routed states: continue-target w_t = sigmoid(k (relu(L_{t-1} - L_t) - gamma)) for the gates
           after loops 2 and 3 (the gate after loop 1 gets no target, as in the paper; it shares the weights)
  next     forward-looking: w_t = sigmoid(k (L_t - L_{t+1} - gamma)), "will the NEXT loop improve the loss", for the gates after
           loops 1, 2 and 3 (needs the token to have run loop t+1)
  correct  exit-target c_t = 1 when the readout after loop t already predicts the full-depth token (gates after loops 1 to 3)

L_t is the cross-entropy of the loop-t readout against the full-depth (teacher) token; k = 50 and gamma = 0.005 as in the paper.
Routing while fitting: the random router of train.py (independent exit loops, per-sequence Dirichlet shares), so the states do
not depend on any of the gates being fitted. Nothing here is reinforcement learning: the targets are supervised labels computed
from the model's own readouts.

  CUDA_VISIBLE_DEVICES=0 python randrouter/stage2.py --tag base [--lora FILE] [--epochs 2] [--init zero|random] [--lr 1e-3]
Writes <OUT>/gates_<tag>/{<variant>.pt ({"weight" (1, 2048), "bias" (1,)}), fit.jsonl, frontier.json (teacher-forced
loops/agreement of every gate at a grid of single thresholds on the validation problems, next to the released gate)}.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
import tf_eval  # noqa: E402

VARIANTS = ("paper", "next", "correct")

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="/home/jli199/boptim_scratch/randrouter")
ap.add_argument("--teacher", default="/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl")
ap.add_argument("--lora", default=None)
ap.add_argument("--lora_r", type=int, default=32)
ap.add_argument("--lora_alpha", type=float, default=64.0)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--init", choices=["zero", "random"], default="zero")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--k", type=float, default=50.0)
ap.add_argument("--gamma", type=float, default=0.005)
ap.add_argument("--budget_tok", type=int, default=6000)
ap.add_argument("--max_len", type=int, default=3072)
ap.add_argument("--train_n", type=int, default=10**9)
ap.add_argument("--val_n", type=int, default=99)
ap.add_argument("--grid", default="0.9,0.7,0.5,0.4,0.3,0.2,0.1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--train_split", default="train")
ap.add_argument("--val_split", default="val")
ap.add_argument("--mem_gb", type=float, default=0.0)
args = ap.parse_args()

run_dir = os.path.join(args.out, f"gates_{args.tag}")
os.makedirs(run_dir, exist_ok=True)
if args.mem_gb:
    torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
torch.manual_seed(args.seed)
gen = torch.Generator().manual_seed(args.seed)
model, tok = core.load_model(core.SERVE_DIR)
if args.lora:
    core.add_lora(model, args.lora_r, args.lora_alpha)
    core.load_lora_state(model, torch.load(args.lora, map_location="cpu"))
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
dim = model.config.hidden_size
gates = {}
for v in VARIANTS:
    g = nn.Linear(dim, 1).cuda().float()
    if args.init == "zero":
        nn.init.zeros_(g.weight)
        nn.init.zeros_(g.bias)
    gates[v] = g
opt = torch.optim.Adam([p for g in gates.values() for p in g.parameters()], lr=args.lr)
train = core.load_teacher(args.teacher, split=args.train_split, max_len=args.max_len)[:args.train_n]
val = core.load_teacher(args.teacher, split=args.val_split, max_len=args.max_len)[:args.val_n]
print(f"gates {VARIANTS} init {args.init}; train {len(train)} sequences, val {len(val)}; lora {args.lora}", flush=True)


def bce(prob, target):
    prob = prob.clamp(1e-6, 1 - 1e-6)
    return -(target * prob.log() + (1 - target) * (1 - prob).log())


step = 0
t0 = time.time()
for epoch in range(args.epochs):
    batches = core.make_batches(train, args.budget_tok, shuffle_gen=gen)
    for batch in batches:
        ids, n_prompt, lengths = core.collate(train, batch, tok.pad_token_id)
        plan, _ = core.sample_iid_plan(ids, gen, alpha=1.0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = core.routed_forward(model, ids, n_prompt, plan=plan)
            pos = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
            read = (pos >= (n_prompt[:, None] - 1)) & (pos < (lengths[:, None] - 1))
            bi, ti = read.nonzero(as_tuple=True)
            target = ids[bi, ti + 1]
            ex = out["exit"][bi, ti]
            feats, ces, oks = [], [], []
            for t in range(core.N_LOOPS):
                hv = out["hidden"][t][bi, ti]
                logits = model.lm_head(hv).float()
                ces.append(F.cross_entropy(logits, target, reduction="none"))
                oks.append((logits.argmax(-1) == target).float())
                feats.append(hv.float())
        losses = {}
        # paper: gates after loops 2 and 3 (0-based t = 1, 2), valid when the token ran loop t
        tot, cnt = 0.0, 0
        for t in (1, 2):
            m = ex >= t
            if m.any():
                w = torch.sigmoid(args.k * (F.relu(ces[t - 1][m] - ces[t][m]) - args.gamma))
                lam = torch.sigmoid(gates["paper"](feats[t][m]).squeeze(-1))
                tot = tot + bce(1 - lam, w).sum()
                cnt += int(m.sum())
        losses["paper"] = tot / max(cnt, 1)
        # next: gates after loops 1 to 3 (t = 0, 1, 2), valid when the token also ran loop t + 1
        tot, cnt = 0.0, 0
        for t in (0, 1, 2):
            m = ex >= t + 1
            if m.any():
                w = torch.sigmoid(args.k * (ces[t][m] - ces[t + 1][m] - args.gamma))
                lam = torch.sigmoid(gates["next"](feats[t][m]).squeeze(-1))
                tot = tot + bce(1 - lam, w).sum()
                cnt += int(m.sum())
        losses["next"] = tot / max(cnt, 1)
        # correct: gates after loops 1 to 3, valid when the token ran loop t
        tot, cnt = 0.0, 0
        for t in (0, 1, 2):
            m = ex >= t
            if m.any():
                lam = torch.sigmoid(gates["correct"](feats[t][m]).squeeze(-1))
                tot = tot + bce(lam, oks[t][m]).sum()
                cnt += int(m.sum())
        losses["correct"] = tot / max(cnt, 1)
        opt.zero_grad(set_to_none=True)
        sum(losses.values()).backward()
        opt.step()
        step += 1
        if step % 10 == 0:
            rec = {"step": step, "epoch": epoch, **{v: float(l) for v, l in losses.items()}, "s": time.time() - t0}
            with open(os.path.join(run_dir, "fit.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"step {step} ep {epoch} " + " ".join(f"{v} {float(l):.4f}" for v, l in losses.items()) + f" {time.time() - t0:.0f}s", flush=True)
        del out, feats, ces, oks

for v, g in gates.items():
    torch.save({"weight": g.weight.detach().cpu(), "bias": g.bias.detach().cpu(), "variant": v, "args": vars(args)}, os.path.join(run_dir, f"{v}.pt"))

# teacher-forced frontier on the validation problems: every gate, and the released gate, at a grid of single thresholds
grid = [float(x) for x in args.grid.split(",")]
routers = [tf_eval.parse_router("full")] + [tf_eval.parse_router(str(q)) for q in grid]
frontier = {}
print("released gate:", flush=True)
frontier["released"] = tf_eval.evaluate(model, val, routers, tok.pad_token_id, args.budget_tok, log=lambda s: print(s, flush=True))
for v, g in gates.items():
    print(f"gate {v}:", flush=True)
    g16 = nn.Linear(dim, 1).cuda().to(torch.bfloat16)
    g16.load_state_dict(g.state_dict())
    frontier[v] = tf_eval.evaluate(model, val, routers[1:], tok.pad_token_id, args.budget_tok, gate=g16, log=lambda s: print(s, flush=True))
json.dump(frontier, open(os.path.join(run_dir, "frontier.json"), "w"), indent=1)
print(f"STAGE2 DONE {args.tag} {time.time() - t0:.0f}s", flush=True)
