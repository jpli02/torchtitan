"""One-off edit (2026-11-20): a loop-1 veto in core.routed_forward and tf_eval.

The released gate decides an exit on the token's own account. The veto is the other half: a read-out of the loop-1 state that
predicts what the exit costs the later tokens (harm_fit.py). A token leaves after loop 1 only if the gate lets it AND its
predicted harm is at most tau; a vetoed token runs loop 2 and is then routed by the gate as usual.
Router spec in tf_eval: "q1,q2,q3|veto=PROBE.pt@tau". Kept for the record; the asserts refuse a second run."""
import pathlib

here = pathlib.Path(__file__).parent
p = here / "core.py"
s = p.read_text()
old = '''def routed_forward(model, ids, n_prompt, *, plan=None, thresholds=None, shares=None, lengths=None, exact_cache=False,
                   grad_ckpt=False, gate=None, fill_maps=None):'''
assert s.count(old) == 1
s = s.replace(old, '''def load_probe(path: str, device: str = "cuda"):
    """A harm probe of harm_fit.py as a function of the normed loop-1 state: (B, T, D) -> (B, T) float32 score."""
    blob = torch.load(path, map_location="cpu")
    sd = blob["state_dict"]
    if blob["kind"] == "mlp":
        net = nn.Sequential(nn.Linear(sd["0.weight"].shape[1], sd["0.weight"].shape[0]), nn.GELU(), nn.Linear(sd["2.weight"].shape[1], 1))
    else:
        net = nn.Linear(sd["weight"].shape[1], 1)
    net.load_state_dict(sd)
    net = net.to(device).float().eval()
    mean, std = blob["mean"].to(device), blob["std"].to(device)

    @torch.no_grad()
    def score(h):
        return net((h.float() - mean) / std).squeeze(-1)

    return score


def routed_forward(model, ids, n_prompt, *, plan=None, thresholds=None, shares=None, lengths=None, exact_cache=False,
                   grad_ckpt=False, gate=None, fill_maps=None, veto=None):''')
old = '''    fill_maps   (key maps, value maps) of load_fill_maps: calibrated fill instead of the plain copy
'''
assert s.count(old) == 1
s = s.replace(old, old + '''    veto        (score function of load_probe, tau): with thresholds, a position leaves after loop 1 only if its score <= tau
''')
old = '''                    else:
                        newly = running & (cum >= float(thresholds[t]))
'''
new = '''                    else:
                        newly = running & (cum >= float(thresholds[t]))
                        if veto is not None and t == 0:
                            newly = newly & (veto[0](h) <= float(veto[1]))
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)

p = here / "tf_eval.py"
s = p.read_text()
old = '''def parse_router(spec):
'''
new = '''_PROBES = {}


def parse_router(spec):
    if "|veto=" in spec:
        base, v = spec.split("|veto=", 1)
        path, tau = v.rsplit("@", 1)
        r = parse_router(base)
        if path not in _PROBES:
            _PROBES[path] = core.load_probe(path)
        return dict(r, spec=spec, veto=(_PROBES[path], float(tau)))
    return dict(_parse_plain(spec), veto=None)


def _parse_plain(spec):
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''                out = core.routed_forward(model, ids, n_prompt, plan=plan, thresholds=r["thr"], shares=r["shares"], lengths=lengths, gate=gate,
                                          fill_maps=fill_maps)
'''
new = '''                out = core.routed_forward(model, ids, n_prompt, plan=plan, thresholds=r["thr"], shares=r["shares"], lengths=lengths, gate=gate,
                                          fill_maps=fill_maps, veto=r.get("veto"))
'''
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace('''leaves; the gate-ordered counterpart of the random router).''',
              '''leaves; the gate-ordered counterpart of the random router). "q1,q2,q3|veto=PROBE.pt@tau": a token leaves after loop 1 only if
the harm probe (harm_fit.py) scores it at most tau.''')
p.write_text(s)
print("patched core.py and tf_eval.py")
