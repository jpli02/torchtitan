"""One-off edit (2026-11-20): calibrated cache fill in core.routed_forward. With `fill_maps` the cache slot of a loop a token
skipped holds an affine map (exit loop -> that loop; one map per layer and head, keys in the unrotated space) of the keys and
values of its exit loop instead of a plain copy: the fill of boptim_lcb/make_exitlab_modeling.py (decode_fill_maps), which kept
BFCL's accuracy at 1.6 loops. Maps come from randrouter/fit_fill_maps.py (or boptim_lcb/kv_fill_fit.py; same file format).
Kept for the record; the asserts refuse a second run."""
import pathlib

here = pathlib.Path(__file__).parent
p = here / "core.py"
s = p.read_text()

old = '''def _block(layer, rot, h, cos, sin, k_prev, v_prev, stale):
    """One decoder layer on the whole sequence; keys/values of the positions in `stale` are replaced by the carried ones."""
'''
new = '''def load_fill_maps(path: str, device: str = "cuda"):
    """Maps of the calibrated fill as ({(exit loop, target loop): [layers, heads, dim + 1, dim]} for keys, the same for values)."""
    blob = torch.load(path, map_location="cpu")
    km, vm = {}, {}
    for name, m in blob["maps"].items():
        w, a, b = (int(x) for x in name.split(":"))
        (km if w == 0 else vm)[(a, b)] = m.to(device, torch.float32)
    return km, vm


def _rot_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _mapped_fill(fill, k_carry, v_carry, cos, sin):
    """Affine stand-ins for this loop's keys/values of the positions that left earlier, from their exit loop's (carried) ones."""
    exit_, t, l, km, vm = fill
    c, s_ = cos[:, None].float(), sin[:, None].float()
    k_un = k_carry.float() * c - _rot_half(k_carry.float()) * s_          # undo the rotary rotation
    v32 = v_carry.float()
    k_out, v_out = k_carry, v_carry
    for e in range(t):
        m = (exit_ == e)[:, None, :, None]
        if not bool(m.any()):
            continue
        a_k, a_v = km[(e, t)][l], vm[(e, t)][l]                            # [heads, dim + 1, dim]
        k_m = torch.einsum("bhtd,hde->bhte", k_un, a_k[:, :-1, :]) + a_k[:, -1, :][None, :, None, :]
        k_m = k_m * c + _rot_half(k_m) * s_
        v_m = torch.einsum("bhtd,hde->bhte", v32, a_v[:, :-1, :]) + a_v[:, -1, :][None, :, None, :]
        k_out = torch.where(m, k_m.to(k_carry.dtype), k_out)
        v_out = torch.where(m, v_m.to(v_carry.dtype), v_out)
    return k_out, v_out


def _block(layer, rot, h, cos, sin, k_prev, v_prev, stale, fill=None):
    """One decoder layer on the whole sequence; keys/values of the positions in `stale` are replaced by the carried ones
    (plain copy of their exit loop's) or, with `fill`, by the calibrated maps of them. Returns the state and the keys/values
    to carry: every position's keys/values of its last computed loop, unmapped."""
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    if k_prev is not None:
        m = stale[:, None, :, None]
        k = torch.where(m, k_prev, k)
        v = torch.where(m, v_prev, v)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
'''
new = '''    k_carry, v_carry = k, v
    if k_prev is not None:
        m = stale[:, None, :, None]
        k_carry = torch.where(m, k_prev, k)
        v_carry = torch.where(m, v_prev, v)
        if fill is None:
            k, v = k_carry, v_carry
        else:
            k_f, v_f = _mapped_fill(fill, k_carry, v_carry, cos, sin)
            k = torch.where(m, k_f, k)
            v = torch.where(m, v_f, v)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    h = h + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))
    return h, k, v
'''
new = '''    h = h + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))
    return h, k_carry, v_carry
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''def routed_forward(model, ids, n_prompt, *, plan=None, thresholds=None, shares=None, lengths=None, exact_cache=False,
                   grad_ckpt=False, gate=None):'''
assert s.count(old) == 1
s = s.replace(old, '''def routed_forward(model, ids, n_prompt, *, plan=None, thresholds=None, shares=None, lengths=None, exact_cache=False,
                   grad_ckpt=False, gate=None, fill_maps=None):''')
old = '''    gate        module mapping the normed state to the gate logit (default: the model's own early_exit_gate)
'''
new = '''    gate        module mapping the normed state to the gate logit (default: the model's own early_exit_gate)
    fill_maps   (key maps, value maps) of load_fill_maps: calibrated fill instead of the plain copy
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''        for l, layer in enumerate(inner.layers):
            kp, vp = (ks[l], vs[l]) if st is not None else (None, None)
            if grad_ckpt:
                h, ks[l], vs[l] = checkpoint(_block, layer, rot, h, cos, sin, kp, vp, st, use_reentrant=False)
            else:
                h, ks[l], vs[l] = _block(layer, rot, h, cos, sin, kp, vp, st)
'''
new = '''        for l, layer in enumerate(inner.layers):
            kp, vp = (ks[l], vs[l]) if st is not None else (None, None)
            fill = (exit_, t, l, fill_maps[0], fill_maps[1]) if (fill_maps is not None and st is not None) else None
            if grad_ckpt:
                h, ks[l], vs[l] = checkpoint(_block, layer, rot, h, cos, sin, kp, vp, st, fill, use_reentrant=False)
            else:
                h, ks[l], vs[l] = _block(layer, rot, h, cos, sin, kp, vp, st, fill)
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)

# tf_eval: --fill_maps
p = here / "tf_eval.py"
s = p.read_text()
old = "def evaluate(model, seqs, routers, pad_id, budget_tok=6000, gate=None, log=print):"
assert s.count(old) == 1
s = s.replace(old, "def evaluate(model, seqs, routers, pad_id, budget_tok=6000, gate=None, log=print, fill_maps=None):")
old = '''                out = core.routed_forward(model, ids, n_prompt, plan=plan, thresholds=r["thr"], shares=r["shares"], lengths=lengths, gate=gate)
'''
new = '''                out = core.routed_forward(model, ids, n_prompt, plan=plan, thresholds=r["thr"], shares=r["shares"], lengths=lengths, gate=gate,
                                          fill_maps=fill_maps)
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    ap.add_argument("--gate", default=None, help="gate file of randrouter/stage2.py (replaces the model's exit gate)")
'''
assert s.count(old) == 1
s = s.replace(old, old + '''    ap.add_argument("--fill_maps", default=None, help="calibrated cache fill (randrouter/fit_fill_maps.py) instead of the plain copy")
    ap.add_argument("--mem_gb", type=float, default=0.0)
''')
old = '''    model, tok = core.load_model(args.ckpt)
    if args.lora:
'''
new = '''    if args.mem_gb:
        torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    model, tok = core.load_model(args.ckpt)
    if args.lora:
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    res = evaluate(model, seqs, [parse_router(s) for s in args.routers.split(";")], tok.pad_token_id, args.budget_tok)
'''
new = '''    fm = core.load_fill_maps(args.fill_maps) if args.fill_maps else None
    res = evaluate(model, seqs, [parse_router(s) for s in args.routers.split(";")], tok.pad_token_id, args.budget_tok, fill_maps=fm)
'''
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace('''json.dump({"ckpt": args.ckpt, "lora": args.lora, "gate": args.gate, "split": args.split,''',
              '''json.dump({"ckpt": args.ckpt, "lora": args.lora, "gate": args.gate, "fill_maps": args.fill_maps, "split": args.split,''')
p.write_text(s)
print("patched core.py and tf_eval.py")
