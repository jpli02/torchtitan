"""One-off edit (2026-11-20): a third router for core.routed_forward, "shares": after loop s the still-running answer tokens
of every sequence are ranked by the gate's cumulative exit probability and the top share f_s of them leaves. It is the
gate-ordered counterpart of the random router (same exit shares, tokens chosen by the gate instead of at random) and, unlike
thresholds drawn uniformly in [0.05, 1], it produces every mixture of exit loops: the released gate's cumulative probability
after loop 1 is squeezed under 0.25, so a uniform threshold either lets almost every token leave after loop 1 or none.
Kept for the record; the asserts refuse a second run."""
import pathlib

here = pathlib.Path(__file__).parent
p = here / "core.py"
s = p.read_text()

old = "def routed_forward(model, ids, n_prompt, *, plan=None, thresholds=None, exact_cache=False, grad_ckpt=False, gate=None):"
assert s.count(old) == 1
s = s.replace(old, "def routed_forward(model, ids, n_prompt, *, plan=None, thresholds=None, shares=None, lengths=None, exact_cache=False,\n"
                   "                   grad_ckpt=False, gate=None):")
old = '''    thresholds  [q1, q2, q3]: a position leaves after loop s (0-based s = 0, 1, 2) when the gate's cumulative exit
'''
new = '''    shares      [f1, f2, f3] (or a (B, 3) tensor): after loop s the top share f_{s+1} of the still-running answer tokens of
                each sequence, ranked by the gate's cumulative exit probability, leaves (needs `lengths`, (B,)); or
    thresholds  [q1, q2, q3]: a position leaves after loop s (0-based s = 0, 1, 2) when the gate's cumulative exit
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''            if thresholds is not None and plan is None:
                with torch.no_grad():
                    cum = cum + lam.detach() * rem
                    rem = rem * (1.0 - lam.detach())
                    newly = is_gen & (exit_ == last) & (cum >= float(thresholds[t]))
                    exit_ = torch.where(newly, torch.full_like(exit_, t), exit_)
'''
new = '''            if (thresholds is not None or shares is not None) and plan is None:
                with torch.no_grad():
                    cum = cum + lam.detach() * rem
                    rem = rem * (1.0 - lam.detach())
                    running = is_gen & (exit_ == last)
                    if shares is not None:
                        running = running & (pos < lengths[:, None])
                        newly = torch.zeros_like(running)
                        for bidx in range(b):
                            idx = running[bidx].nonzero(as_tuple=True)[0]
                            f = float(shares[bidx][t]) if torch.is_tensor(shares) else float(shares[t])
                            k = int(round(f * idx.numel()))
                            if k > 0:
                                top = torch.topk(cum[bidx, idx], k).indices
                                newly[bidx, idx[top]] = True
                    else:
                        newly = running & (cum >= float(thresholds[t]))
                    exit_ = torch.where(newly, torch.full_like(exit_, t), exit_)
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''def sample_thresholds(gen: torch.Generator, lo: float = 0.05, hi: float = 1.0):'''
new = '''def sample_shares(b: int, gen: torch.Generator, alpha: float = 1.0):
    """Per-sequence exit shares for the "shares" router: pi ~ Dirichlet(alpha) over the 4 exit loops, turned into the share of
    the still-running tokens that leaves after loops 1, 2 and 3 (so the realised exit distribution is pi)."""
    state = torch.random.get_rng_state()
    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
    x = torch.distributions.Gamma(torch.full((b, N_LOOPS), float(alpha)), torch.ones(b, N_LOOPS)).sample()
    torch.random.set_rng_state(state)
    pi = x / x.sum(-1, keepdim=True)
    left = torch.ones(b)
    f = []
    for t in range(N_LOOPS - 1):
        f.append((pi[:, t] / left.clamp(min=1e-6)).clamp(0, 1))
        left = left - pi[:, t]
    return torch.stack(f, dim=-1), pi


def sample_thresholds(gen: torch.Generator, lo: float = 0.05, hi: float = 1.0):'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)

# tf_eval: "rank:f1,f2,f3" router
p = here / "tf_eval.py"
s = p.read_text()
old = '''    if spec.startswith("iid:"):
        return {"spec": spec, "thr": None, "rates": [float(v) for v in spec[4:].split(",")]}
    v = [float(x) for x in spec.split(",")]
    return {"spec": spec, "thr": v * 3 if len(v) == 1 else v, "rates": None}
'''
new = '''    if spec.startswith("iid:"):
        return {"spec": spec, "thr": None, "rates": [float(v) for v in spec[4:].split(",")], "shares": None}
    if spec.startswith("rank:"):
        return {"spec": spec, "thr": None, "rates": None, "shares": [float(v) for v in spec[5:].split(",")]}
    v = [float(x) for x in spec.split(",")]
    return {"spec": spec, "thr": v * 3 if len(v) == 1 else v, "rates": None, "shares": None}
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''        return {"spec": spec, "thr": None, "rates": None}
'''
assert s.count(old) == 1
s = s.replace(old, '''        return {"spec": spec, "thr": None, "rates": None, "shares": None}
''')
old = '''                out = core.routed_forward(model, ids, n_prompt, plan=plan, thresholds=r["thr"], gate=gate)
'''
new = '''                out = core.routed_forward(model, ids, n_prompt, plan=plan, thresholds=r["thr"], shares=r["shares"], lengths=lengths, gate=gate)
'''
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace('''A router is "full", one threshold q (the released single knob), "q1,q2,q3", or "iid:p1,p2,p3,p4" (the random router with those
exit shares, seed 0).''', '''A router is "full", one threshold q (the released single knob), "q1,q2,q3", "iid:p1,p2,p3,p4" (the random router with those
exit shares, seed 0), or "rank:f1,f2,f3" (after loop s the share f_s of the still-running tokens that the gate is surest about
leaves; the gate-ordered counterpart of the random router).''')
p.write_text(s)

# train.py: the shares router in the mixture
p = here / "train.py"
s = p.read_text()
old = '''ap.add_argument("--p_iid", type=float, default=0.6)
ap.add_argument("--p_gate", type=float, default=0.3)
'''
new = '''ap.add_argument("--p_iid", type=float, default=0.4)
ap.add_argument("--p_rank", type=float, default=0.4)
ap.add_argument("--p_gate", type=float, default=0.1)
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''        plan = thr = None
        if u < args.p_iid:
            kind = "iid"
            plan, _ = core.sample_iid_plan(ids, gen, alpha=args.dir_alpha)
        elif u < args.p_iid + args.p_gate:
            kind = "gate"
            thr = core.sample_thresholds(gen)
        else:
            kind = "full"
'''
new = '''        plan = thr = shares = None
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
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''            out = core.routed_forward(model, ids, n_prompt, plan=plan, thresholds=thr, grad_ckpt=not args.no_ckpt)
'''
new = '''            out = core.routed_forward(model, ids, n_prompt, plan=plan, thresholds=thr, shares=shares, lengths=lengths, grad_ckpt=not args.no_ckpt)
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''  * iid   (share --p_iid)  the random router: every answer token draws its exit loop independently from a per-sequence
                           distribution over the 4 loops, itself drawn from Dirichlet(--dir_alpha);
  * gate  (share --p_gate) the gate with random per-loop thresholds, uniform in the router searches' box [0.05, 1]^3;
'''
new = '''  * iid   (share --p_iid)  the random router: every answer token draws its exit loop independently from a per-sequence
                           distribution over the 4 loops, itself drawn from Dirichlet(--dir_alpha);
  * rank  (share --p_rank) the same random exit shares, but the tokens that leave after a loop are the ones the released gate
                           is surest about (what a threshold router does, at every mixture of exit loops);
  * gate  (share --p_gate) the gate with random per-loop thresholds, uniform in the router searches' box [0.05, 1]^3;
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''ap.add_argument("--val_routers", default="full;0.5;0.3;0.2;iid:0.25,0.25,0.25,0.25;iid:0.1,0.4,0.4,0.1")'''
assert s.count(old) == 1
s = s.replace(old, '''ap.add_argument("--val_routers", default="full;0.3;0.245,0.3,0.3;0.2;0.25,1.0,1.0;rank:0.5,0.5,0.5;iid:0.25,0.25,0.25,0.25")''')
p.write_text(s)
print("patched core.py, tf_eval.py, train.py")
