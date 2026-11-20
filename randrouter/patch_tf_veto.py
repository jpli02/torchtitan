"""One-off edit (2026-11-20): the loop-1 veto as a fourth search variable of the proxy objective.

ouro_tf_router.py: veto=PROBE.pt adds the variable tau in [-2.5, 3.0] (the probe's score scale; 3.0 vetoes nothing): a token
leaves after loop 1 only if the gate lets it and the harm probe scores it at most tau.
setup_tf_objective.py / tf_bo_prep.py: the --ouro_tf_veto flag and --veto option; the single-knob warm start gets tau = 3.0.
Kept for the record; the asserts refuse a second run."""
import pathlib

here = pathlib.Path(__file__).parent
p = here / "ouro_tf_router.py"
s = p.read_text()
old = "    def __init__(self, *, lora: str | None = None, gate: str | None = None, fill_maps: str | None = None,\n"
assert s.count(old) == 1
s = s.replace(old, "    def __init__(self, *, lora: str | None = None, gate: str | None = None, fill_maps: str | None = None, veto: str | None = None,\n")
old = '''        self.params = [ObjectiveParam("q1", (0.05, 1.0)), ObjectiveParam("q2", (0.05, 1.0)), ObjectiveParam("q3", (0.05, 1.0))]
'''
new = '''        self.params = [ObjectiveParam("q1", (0.05, 1.0)), ObjectiveParam("q2", (0.05, 1.0)), ObjectiveParam("q3", (0.05, 1.0))]
        self.veto = veto
        if veto:   # fourth variable: the loop-1 veto threshold on the harm probe's score (3.0 vetoes nothing)
            self.params.append(ObjectiveParam("tau", (-2.5, 3.0)))
            self.setup["veto"] = veto
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    def measure(self, thr):
        """(loops, agreement %) of the router with thresholds thr on the held-out answers, with the decoder's cache."""
        spec = ",".join(f"{float(v):.6f}" for v in thr)
'''
new = '''    def measure(self, thr, tau=None):
        """(loops, agreement %) of the router with thresholds thr on the held-out answers, with the decoder's cache."""
        spec = ",".join(f"{float(v):.6f}" for v in thr)
        if self.veto and tau is not None:
            spec += f"|veto={self.veto}@{float(tau):.6f}"
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''        thr = [min(1.0, max(0.05, float(v))) for v in params][:3]
        t0 = time.time()
        loops, agree, share = self.measure(thr)
        value = -6.25 * 4.0 / loops + self.penalty * max(0.0, self.floor - agree)
        rec = {"thresholds": thr, "loops": loops,'''
new = '''        thr = [min(1.0, max(0.05, float(v))) for v in params][:3]
        tau = float(params[3]) if self.veto and len(params) > 3 else None
        t0 = time.time()
        loops, agree, share = self.measure(thr, tau)
        value = -6.25 * 4.0 / loops + self.penalty * max(0.0, self.floor - agree)
        rec = {"thresholds": thr, "tau": tau, "loops": loops,'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''        print(f"[tf-router] thr={['%.3f' % v for v in thr]} loops={loops:.3f}'''
assert s.count(old) == 1
s = s.replace(old, '''        print(f"[tf-router] thr={['%.3f' % v for v in thr]}{'' if tau is None else ' tau=%.3f' % tau} loops={loops:.3f}''')
p.write_text(s)

p = here / "setup_tf_objective.py"
s = p.read_text()
old = '''else:
    print("objective/__init__.py already has ouro_tf_router")
'''
new = '''else:
    print("objective/__init__.py already has ouro_tf_router")
s = open(p).read()
if "ouro_tf_veto" not in s:
    a = '            fill_maps=getattr(args, "ouro_tf_fill_maps", None),\\n'
    assert s.count(a) == 1
    s = s.replace(a, a + '            veto=getattr(args, "ouro_tf_veto", None),\\n', 1)
    open(p, "w").write(s)
    py_compile.compile(p, doraise=True)
    print("added veto to the ouro_tf_router factory branch")
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''else:
    print("main.py already has the --ouro_tf_* flags")
'''
new = '''else:
    print("main.py already has the --ouro_tf_* flags")
s = open(p).read()
if "--ouro_tf_veto" not in s:
    a = '    parser.add_argument("--ouro_tf_fill_maps", type=str, default=None, help="[ouro_tf_router] calibrated cache fill")\\n'
    assert s.count(a) == 1
    s = s.replace(a, a + '    parser.add_argument("--ouro_tf_veto", type=str, default=None, help="[ouro_tf_router] harm probe: adds the loop-1 veto threshold as a fourth variable")\\n', 1)
    open(p, "w").write(s)
    py_compile.compile(p, doraise=True)
    print("added the --ouro_tf_veto flag to main.py")
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)

p = here / "tf_bo_prep.py"
s = p.read_text()
old = 'ap.add_argument("--gate", default=None)\n'
assert s.count(old) == 1
s = s.replace(old, old + 'ap.add_argument("--veto", default=None, help="harm probe: the loop-1 veto threshold becomes a fourth search variable")\n')
old = '''obj = OuroTFRouterObjective(lora=args.lora, gate=args.gate, fill_maps=args.fill_maps, n_items=args.n, floor_pct=args.floor,'''
assert s.count(old) == 1
s = s.replace(old, '''obj = OuroTFRouterObjective(lora=args.lora, gate=args.gate, fill_maps=args.fill_maps, veto=args.veto, n_items=args.n, floor_pct=args.floor,''')
old = '''    v = obj.evaluate([q, q, q])
    rec = obj._eval_records[-1]
    records.append({"x": [q, q, q], "fval": v})
'''
new = '''    x = [q, q, q] + ([3.0] if args.veto else [])        # 3.0 = the veto never fires
    v = obj.evaluate(x)
    rec = obj._eval_records[-1]
    records.append({"x": x, "fval": v})
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''for k, v in (("ouro_tf_fill_maps", args.fill_maps), ("ouro_tf_lora", args.lora), ("ouro_tf_gate", args.gate)):'''
assert s.count(old) == 1
s = s.replace(old, '''for k, v in (("ouro_tf_fill_maps", args.fill_maps), ("ouro_tf_lora", args.lora), ("ouro_tf_gate", args.gate), ("ouro_tf_veto", args.veto)):''')
p.write_text(s)
print("patched ouro_tf_router.py, setup_tf_objective.py, tf_bo_prep.py")
