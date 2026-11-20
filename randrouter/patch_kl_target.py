"""One-off edit (2026-11-20): a distillation target for the random-router training.

The first run (hard labels: cross-entropy to the full-depth token at every readout) did learn the cache (after 125 steps the
readouts under the decoder's cache agree with the exact-cache ones within 0.5 point, from 2 to 4 points apart), but it also
moved the model: full-depth agreement with the base model's own answers fell from 99.76 % to 97.97 % on held-out problems
while the loss on them fell, i.e. the model got sharper on about 100 training answers and changed its mind elsewhere.

New target (train.py --target kl): for every valid readout (position i, loop t) under the router's cache, the KL divergence from
the BASE model's readout of the same position with an exact cache, at the same loop t (--kl_teacher same) or at the last loop
(--kl_teacher final). The base model is the same network with the LoRA updates switched off (one extra forward without
gradients, no second copy of the weights). With the same-loop teacher the loss is zero wherever the cache does not matter and
asks for nothing but "the filled cache slots must not change the readout": no pressure to sharpen, to improve an early loop or
to move the full-depth model.
Kept for the record; the asserts refuse a second run."""
import pathlib

here = pathlib.Path(__file__).parent
p = here / "core.py"
s = p.read_text()

old = '''    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale
'''
new = '''    enabled = True  # class-wide switch: False = the base model (teacher pass of the distillation target)

    def forward(self, x):
        if not LoRALinear.enabled:
            return self.base(x)
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale
'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''def _head_ce(head, x, tg):
    logits = head(x).float()
    return F.cross_entropy(logits, tg, reduction="none"), (logits.argmax(-1) == tg).float()
'''
new = '''def _head_ce(head, x, tg):
    logits = head(x).float()
    return F.cross_entropy(logits, tg, reduction="none"), (logits.argmax(-1) == tg).float()


def _head_kl(head, x, x_teacher, tg):
    """KL(teacher readout || student readout) per position, and whether the student's top token is the teacher-forced one."""
    logits = head(x).float()
    logp = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        logp_t = F.log_softmax(head(x_teacher).float(), dim=-1)
        p_t = logp_t.exp()
    return (p_t * (logp_t - logp)).sum(-1), (logits.argmax(-1) == tg).float()


@torch.no_grad()
def teacher_hidden(model, ids, n_prompt):
    """Normed states of the base model (LoRA off) at full depth with an exact cache: [4 x (B, T, D)]."""
    was = LoRALinear.enabled
    LoRALinear.enabled = False
    try:
        out = routed_forward(model, ids, n_prompt)
    finally:
        LoRALinear.enabled = was
    return [h.detach() for h in out["hidden"]]
'''
assert s.count(old) == 1
s = s.replace(old, new)

old = 'def readout_stats(model, out, ids, n_prompt, lengths, *, loss_on="valid", need_grad=True, loop_w=None):'
assert s.count(old) == 1
s = s.replace(old, 'def readout_stats(model, out, ids, n_prompt, lengths, *, loss_on="valid", need_grad=True, loop_w=None, teacher=None,\n'
                   '                  kl_teacher="same"):')
old = '''    loop_w  per-loop weights of the "valid" loss (default 1, 1, 1, 1).'''
new = '''    teacher  states of teacher_hidden(): the loss of a readout is then the KL divergence from the base model's exact-cache
             readout of the same position at the same loop (kl_teacher="same") or at the last loop ("final"), instead of the
             cross-entropy to the token; the reported "ce" sums are that KL, agreement is still against the token.
    loop_w  per-loop weights of the "valid" loss (default 1, 1, 1, 1).'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''        if need_grad and hv.requires_grad:
            # the vocabulary-sized logits of a loop are recomputed in the backward pass instead of being kept for all loops
            # (the targets are passed as an argument: the function runs again in the backward pass, after this loop has moved on)
            ce, ok = checkpoint(_head_ce, model.lm_head, hv, tg, use_reentrant=False)
        else:
            ce, ok = _head_ce(model.lm_head, hv, tg)
'''
new = '''        if teacher is not None:
            ht = teacher[t if kl_teacher == "same" else N_LOOPS - 1][bi[valid], ti[valid]]
            if need_grad and hv.requires_grad:
                ce, ok = checkpoint(_head_kl, model.lm_head, hv, ht, tg, use_reentrant=False)
            else:
                ce, ok = _head_kl(model.lm_head, hv, ht, tg)
        elif need_grad and hv.requires_grad:
            # the vocabulary-sized logits of a loop are recomputed in the backward pass instead of being kept for all loops
            # (the targets are passed as an argument: the function runs again in the backward pass, after this loop has moved on)
            ce, ok = checkpoint(_head_ce, model.lm_head, hv, tg, use_reentrant=False)
        else:
            ce, ok = _head_ce(model.lm_head, hv, tg)
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)

p = here / "train.py"
s = p.read_text()
old = 'ap.add_argument("--loss_on", choices=["valid", "exit"], default="valid")\n'
assert s.count(old) == 1
s = s.replace(old, old + 'ap.add_argument("--target", choices=["kl", "ce"], default="kl", help="kl: distil the base model\'s exact-cache readouts; ce: cross-entropy to the token")\n'
                         'ap.add_argument("--kl_teacher", choices=["same", "final"], default="same")\n'
                         'ap.add_argument("--lora_targets", default=",".join(core.LORA_TARGETS), help="projections that get a LoRA update")\n')
old = '''        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = core.routed_forward(model, ids, n_prompt, plan=plan, thresholds=thr, shares=shares, lengths=lengths, grad_ckpt=not args.no_ckpt,
                                      fill_maps=FILL)
            loss, st = core.readout_stats(model, out, ids, n_prompt, lengths, loss_on=args.loss_on, loop_w=LOOP_W)
'''
new = '''        with torch.autocast("cuda", dtype=torch.bfloat16):
            teach = core.teacher_hidden(model, ids, n_prompt) if args.target == "kl" else None
            out = core.routed_forward(model, ids, n_prompt, plan=plan, thresholds=thr, shares=shares, lengths=lengths, grad_ckpt=not args.no_ckpt,
                                      fill_maps=FILL)
            loss, st = core.readout_stats(model, out, ids, n_prompt, lengths, loss_on=args.loss_on, loop_w=LOOP_W, teacher=teach,
                                          kl_teacher=args.kl_teacher)
        del teach
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = "params = core.add_lora(model, args.r, args.alpha)\n"
assert s.count(old) == 1
s = s.replace(old, 'params = core.add_lora(model, args.r, args.alpha, targets=tuple(args.lora_targets.split(",")))\n')
old = '''with the decoder's cache fill (core.routed_forward), and minimises the cross-entropy between the full-depth token and the
readout of every loop the token ran (--loss_on valid) or of its exit loop only (--loss_on exit).
'''
new = '''with the decoder's cache fill (core.routed_forward). Loss, over the readouts of every loop a token ran (--loss_on valid) or of
its exit loop only (--loss_on exit):
  --target kl (default)  KL divergence from the base model's readout of the same position and loop with an EXACT cache (the
                         same network with the LoRA updates off): "a filled cache slot must not change the readout";
  --target ce            cross-entropy to the full-depth token (the first run; it learned the cache but moved the model:
                         full-depth agreement with the base answers fell from 99.8 to 98.0 % on held-out problems).
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print("patched core.py and train.py")
