"""One-off edit (2026-11-20): the proxy objective on an EXACT cache (exact=1): the router's exits with every token's exact keys
and values, i.e. what a search could reach if the cache of early exits were repaired (or the skipped loops were computed in
the background). ouro_tf_router.py, setup_tf_objective.py (--ouro_tf_exact), tf_bo_prep.py (--exact).
Kept for the record; the asserts refuse a second run."""
import pathlib

here = pathlib.Path(__file__).parent
p = here / "ouro_tf_router.py"
s = p.read_text()
old = "                 work_dir: str | None = None, lora_r: int = 32, lora_alpha: float = 64.0, seed: int | None = None):"
assert s.count(old) == 1
s = s.replace(old, "                 work_dir: str | None = None, lora_r: int = 32, lora_alpha: float = 64.0, seed: int | None = None, exact: int = 0):")
old = '''        self.seed = seed
        self._eval_records: list = []
'''
new = '''        self.seed = seed
        self.exact = bool(int(exact))
        self.setup["exact_cache"] = self.exact
        self._eval_records: list = []
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''                                log=lambda s: None, fill_maps=self.fill, exact=False)[spec]
        return res["loops"], 100.0 * res["agree_real"], res["share"]
'''
new = '''                                log=lambda s: None, fill_maps=self.fill, exact=self.exact)[spec]
        return res["loops"], 100.0 * res["agree_exact" if self.exact else "agree_real"], res["share"]
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)

p = here / "setup_tf_objective.py"
s = p.read_text()
old = '''    print("added veto to the ouro_tf_router factory branch")
'''
new = '''    print("added veto to the ouro_tf_router factory branch")
s = open(p).read()
if "ouro_tf_exact" not in s:
    a = '            veto=getattr(args, "ouro_tf_veto", None),\\n'
    assert s.count(a) == 1
    s = s.replace(a, a + '            exact=int(getattr(args, "ouro_tf_exact", 0) or 0),\\n', 1)
    open(p, "w").write(s)
    py_compile.compile(p, doraise=True)
    print("added exact to the ouro_tf_router factory branch")
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    print("added the --ouro_tf_veto flag to main.py")
'''
new = '''    print("added the --ouro_tf_veto flag to main.py")
s = open(p).read()
if "--ouro_tf_exact" not in s:
    a = '    parser.add_argument("--ouro_tf_fill_maps", type=str, default=None, help="[ouro_tf_router] calibrated cache fill")\\n'
    assert s.count(a) == 1
    s = s.replace(a, a + '    parser.add_argument("--ouro_tf_exact", type=int, default=0, help="[ouro_tf_router] 1 = exact cache (upper bound of a cache repair)")\\n', 1)
    open(p, "w").write(s)
    py_compile.compile(p, doraise=True)
    print("added the --ouro_tf_exact flag to main.py")
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)

p = here / "tf_bo_prep.py"
s = p.read_text()
old = 'ap.add_argument("--gate", default=None)\n'
assert s.count(old) == 1
s = s.replace(old, old + 'ap.add_argument("--exact", action="store_true", help="exact cache: the upper bound of a cache repair")\n')
old = '''obj = OuroTFRouterObjective(lora=args.lora, gate=args.gate, fill_maps=args.fill_maps, veto=args.veto, n_items=args.n, floor_pct=args.floor,'''
assert s.count(old) == 1
s = s.replace(old, '''obj = OuroTFRouterObjective(lora=args.lora, gate=args.gate, fill_maps=args.fill_maps, veto=args.veto, exact=int(args.exact), n_items=args.n,
                            floor_pct=args.floor,''')
old = '''if args.mem_gb:
    objective["ouro_tf_mem_gb"] = args.mem_gb
'''
new = '''if args.mem_gb:
    objective["ouro_tf_mem_gb"] = args.mem_gb
if args.exact:
    objective["ouro_tf_exact"] = 1
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print("patched ouro_tf_router.py, setup_tf_objective.py, tf_bo_prep.py")
