"""One-off edit (2026-11-20) of harm.py: integrated gradients instead of one gradient.

The plain gradient of the damage at a random release set has both signs and a mean near zero (the damage is a KL divergence,
flat at "nothing released" and curved beyond): it is not an attribution. Integrated along the straight path from "nothing
released" (damage 0) to the drawn set, the gradients split that draw's total damage exactly among the released tokens
(completeness: the shares sum to the damage), which is the quantity wanted: how much of the later tokens' damage is due to
releasing token j, given the others released in that draw. Midpoint rule with --ig_steps points.
Kept for the record; the asserts refuse a second run."""
import pathlib

p = pathlib.Path(__file__).parent / "harm.py"
s = p.read_text()
old = '''    for d in range(args.draws):
        m0 = (torch.rand(ids.shape, generator=gen) < args.p).to(ids.device) & is_ans
        mix = m0.float().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hf = mixed_final(ids, mix)
            keep = is_ans & ~m0                                                    # the tokens that pay: not released themselves
            bi, ti = keep.nonzero(as_tuple=True)
            total = ids.new_zeros((), dtype=torch.float32)
            for s in range(0, bi.numel(), 512):                                   # vocabulary logits in slices
                sl = slice(s, s + 512)
                kl, _ = checkpoint(core._head_kl, model.lm_head, hf[bi[sl], ti[sl]], exact["hidden"][-1][bi[sl], ti[sl]], ids[bi[sl], ti[sl] + 1],
                                   use_reentrant=False)
                total = total + kl.sum()
        total.backward()
        g = mix.grad.detach().float()
        harm += g
        harm1 += g * m0.float()
        cnt1 += m0.float()
        dmg += float(total) / max(int(keep.sum()), 1)
        del hf, total, mix
'''
new = '''    for d in range(args.draws):
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
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = 'ap.add_argument("--p", type=float, default=0.4)\n'
assert s.count(old) == 1
s = s.replace(old, old + 'ap.add_argument("--ig_steps", type=int, default=4, help="points of the integrated-gradients path per draw")\n')
old = '''rows = []
t0 = time.time()
'''
new = '''rows = []
ig_sum = dmg_sum = 0.0
t0 = time.time()
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''        print(f"batch {bi_ + 1}/{len(batches)}  mean damage per paying token {dmg / args.draws:.4f} nats  {time.time() - t0:.0f}s "
              f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
'''
new = '''        print(f"batch {bi_ + 1}/{len(batches)}  mean damage per paying token {dmg / args.draws:.4f} nats; attributed / measured damage so far "
              f"{ig_sum / max(dmg_sum, 1e-9):.3f}  {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
'''
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace('''  * dD/dm_j from one backward pass is, to first order, what releasing (m_j = 0 -> 1) or keeping (1 -> 0) token j changes in the
    damage of all later tokens; averaged over a few random sets it is the token's "harm".''',
              '''  * the gradient dD/dm_j, integrated along the path from "nothing released" (D = 0) to the drawn set (integrated gradients), splits
    that draw's damage exactly among the released tokens; a token's share, averaged over the draws in which it was released
    ("harm_released"), is its harm: what its early exit costs the later tokens.''')
p.write_text(s)
print("patched harm.py")
