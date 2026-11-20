"""What does a token's early exit cost the LATER tokens? (The signal the released gate does not have.)

The gate is trained on the token's own loss (Stage II: "did this loop still improve my prediction"), and by that measure most
tokens are safe to release after loop 1. But a released token leaves stand-ins of its loop-1 keys and values in the cache
slots of loops 2 to 4, and the later tokens read them: the cost of an exit is paid by OTHER tokens. Under a random router this
externality can be measured for every token at once, without any reward or policy learning:

  * draw a random set of answer tokens to release after loop 1 (probability --p each); every token still computes all loops;
  * in loops 2 to 4 the keys/values other tokens see for token j are  fresh_j + m_j * (stand_in_j - fresh_j),  with m_j = 1 for
    the released tokens and 0 otherwise (stand-in = the copy of the loop-1 keys/values, or their calibrated map);
  * damage D = sum over the NOT released tokens of KL(base readout with an exact cache || readout with this cache), last loop;
  * the gradient dD/dm_j, integrated along the path from "nothing released" (D = 0) to the drawn set (integrated gradients), splits
    that draw's damage exactly among the released tokens; a token's share, averaged over the draws in which it was released
    ("harm_released"), is its harm: what its early exit costs the later tokens.

Written per answer token: harm, the loop-1 state (what a gate could read), the released gate's loop-1 exit probability, whether
the loop-1 readout is already the full-depth token, and the token id. fit: randrouter/harm_fit.py.

  CUDA_VISIBLE_DEVICES=0 python randrouter/harm.py --out FILE.pt [--split train] [--draws 4] [--p 0.4] [--fill_maps F] [--n N]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl")
ap.add_argument("--split", default="train")
ap.add_argument("--n", type=int, default=10**9)
ap.add_argument("--draws", type=int, default=4)
ap.add_argument("--p", type=float, default=0.4)
ap.add_argument("--ig_steps", type=int, default=4, help="points of the integrated-gradients path per draw")
ap.add_argument("--fill_maps", default=None)
ap.add_argument("--budget_tok", type=int, default=2000)
ap.add_argument("--max_len", type=int, default=2000)
ap.add_argument("--mem_gb", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

if args.mem_gb:
    torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
model, tok = core.load_model(core.SERVE_DIR)
for p_ in model.parameters():
    p_.requires_grad_(False)
FILL = core.load_fill_maps(args.fill_maps) if args.fill_maps else None
seqs = core.load_teacher(args.teacher, split=args.split, max_len=args.max_len)[:args.n]
gen = torch.Generator().manual_seed(args.seed)
inner = model.model
rot = core._rotary(model)
N_LAYERS = len(inner.layers)


def _mix_block(layer, h, cos, sin, k1, v1, mix, t, l):
    """A decoder layer of loop t >= 1 in which the keys/values OTHER positions see are mixed towards the loop-1 stand-ins."""
    attn = layer.self_attn
    b, n, _ = h.shape
    x = layer.input_layernorm(h)
    shape = (b, n, -1, attn.head_dim)
    q = attn.q_proj(x).view(shape).transpose(1, 2)
    k = attn.k_proj(x).view(shape).transpose(1, 2)
    v = attn.v_proj(x).view(shape).transpose(1, 2)
    q, k = rot(q, k, cos, sin)
    if FILL is None:
        ks, vs = k1, v1
    else:
        zero = torch.zeros(k.shape[0], k.shape[2], dtype=torch.long, device=k.device)   # every stand-in comes from loop 1
        ks, vs = core._mapped_fill((zero, t, l, FILL[0], FILL[1]), k1, v1, cos, sin)
    m = mix[:, None, :, None].to(k.dtype)
    k = k + m * (ks - k)
    v = v + m * (vs - v)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    a = attn.o_proj(a.transpose(1, 2).reshape(b, n, -1))
    h = h + layer.input_layernorm_2(a)
    h = h + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))
    return h


def mixed_final(ids, mix):
    """Last-loop normed states when the cache of loops 2-4 is mixed towards the loop-1 stand-ins by `mix` (B, T)."""
    h = inner.embed_tokens(ids)
    pos = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
    cos, sin = inner.rotary_emb(h, pos)
    k1, v1 = [], []
    with torch.no_grad():
        for layer in inner.layers:
            h, k, v = core._block(layer, rot, h, cos, sin, None, None, None)
            k1.append(k)
            v1.append(v)
        h = inner.norm(h)
    for t in range(1, core.N_LOOPS):
        for l, layer in enumerate(inner.layers):
            h = checkpoint(_mix_block, layer, h, cos, sin, k1[l], v1[l], mix, t, l, use_reentrant=False)
        h = inner.norm(h)
    return h


rows = []
ig_sum = dmg_sum = 0.0
t0 = time.time()
batches = core.make_batches(seqs, args.budget_tok, max_batch=4)
for bi_, batch in enumerate(batches):
    ids, n_prompt, lengths = core.collate(seqs, batch, tok.pad_token_id)
    pos = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
    is_ans = (pos >= n_prompt[:, None]) & (pos < (lengths[:, None] - 1))          # decoded positions with a next token
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        exact = core.routed_forward(model, ids, n_prompt)
        logp_t = None
    harm = torch.zeros(ids.shape, dtype=torch.float32, device=ids.device)
    cnt1 = torch.zeros_like(harm)
    harm1 = torch.zeros_like(harm)
    dmg = 0.0
    for d in range(args.draws):
        m0 = (torch.rand(ids.shape, generator=gen) < args.p).to(ids.device) & is_ans
        keep = is_ans & ~m0                                                        # the tokens that pay: not released themselves
        bi, ti = keep.nonzero(as_tuple=True)
        ig = torch.zeros_like(harm)
        for a in [(i + 0.5) / args.ig_steps for i in range(args.ig_steps)] + [1.0]:   # midpoints of the path, then its end (the damage itself)
            mix = (a * m0.float()).requires_grad_(a < 1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(a < 1.0):
                hf = mixed_final(ids, mix)
                total = ids.new_zeros((), dtype=torch.float32)
                for s in range(0, bi.numel(), 512):                               # vocabulary logits in slices
                    sl = slice(s, s + 512)
                    if a < 1.0:
                        kl, _ = checkpoint(core._head_kl, model.lm_head, hf[bi[sl], ti[sl]], exact["hidden"][-1][bi[sl], ti[sl]],
                                           ids[bi[sl], ti[sl] + 1], use_reentrant=False)
                    else:
                        kl, _ = core._head_kl(model.lm_head, hf[bi[sl], ti[sl]], exact["hidden"][-1][bi[sl], ti[sl]], ids[bi[sl], ti[sl] + 1])
                    total = total + kl.sum()
            if a < 1.0:
                total.backward()
                ig += mix.grad.detach().float() * m0.float() / args.ig_steps
            else:
                end_damage = float(total)
            del hf, total, mix
        harm += ig                                   # each released token's share of this draw's damage (0 for the others)
        harm1 += ig
        cnt1 += m0.float()
        dmg += end_damage / max(int(keep.sum()), 1)
        ig_sum += float(ig.sum())
        dmg_sum += end_damage
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        bi, ti = is_ans.nonzero(as_tuple=True)
        h1 = exact["hidden"][0][bi, ti]
        ok1 = (model.lm_head(h1).float().argmax(-1) == ids[bi, ti + 1])
        okf = (model.lm_head(exact["hidden"][-1][bi, ti]).float().argmax(-1) == ids[bi, ti + 1])
        rows.append({"seq": torch.tensor([batch[b] for b in bi.tolist()]), "pos": (ti - n_prompt[bi]).cpu(), "token": ids[bi, ti].cpu(),
                     "harm": (harm[bi, ti] / args.draws).cpu(), "harm_released": (harm1[bi, ti] / cnt1[bi, ti].clamp(min=1)).cpu(),
                     "n_released": cnt1[bi, ti].cpu(), "state1": h1.float().half().cpu(), "lam1": exact["lam"][bi, ti, 0].cpu(),
                     "ok1": ok1.cpu(), "ok_full": okf.cpu()})
    del exact
    if (bi_ + 1) % 5 == 0 or bi_ + 1 == len(batches):
        print(f"batch {bi_ + 1}/{len(batches)}  mean damage per paying token {dmg / args.draws:.4f} nats; attributed / measured damage so far "
              f"{ig_sum / max(dmg_sum, 1e-9):.3f}  {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
out = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
out["question_ids"] = [s["question_id"] for s in seqs]
out["args"] = vars(args)
torch.save(out, args.out)
print(f"WROTE {args.out}: {out['harm'].numel()} answer tokens of {len(seqs)} answers", flush=True)
