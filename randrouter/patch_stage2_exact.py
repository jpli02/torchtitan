"""One-off edit (2026-11-20) of stage2.py: fit the gates on exact-cache states (--exact) or under the calibrated fill
(--fill_maps), and compare gates under gate-ranked exit shares instead of thresholds (the fitted gates' probabilities live on
other scales than the released gate's, whose cumulative probability is squeezed under 0.25 / 0.5 / 0.75).
Kept for the record; the asserts refuse a second run."""
import pathlib

p = pathlib.Path(__file__).parent / "stage2.py"
s = p.read_text()
old = 'ap.add_argument("--grid", default="0.9,0.7,0.5,0.4,0.3,0.2,0.1")\n'
assert s.count(old) == 1
s = s.replace(old, 'ap.add_argument("--routers", default="rank:0.5,0.5,0.5;rank:0.7,0.5,0.5;rank:0.8,0.5,0.5;rank:0.9,0.5,0.5;rank:0.8,0.8,0.8;rank:0.9,0.9,0.9",\n'
                   '                help="routers of the frontier: gate-ranked exit shares, so gates with different probability scales compare at equal loops")\n'
                   'ap.add_argument("--exact", action="store_true", help="fit on exact-cache states (the plain full-depth forward): the gate a repaired cache would need")\n'
                   'ap.add_argument("--fill_maps", default=None, help="calibrated cache fill while fitting and in the frontier")\n')
old = '''            out = core.routed_forward(model, ids, n_prompt, plan=plan)
'''
new = '''            out = core.routed_forward(model, ids, n_prompt, plan=plan, exact_cache=args.exact, fill_maps=FILL)
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''            ex = out["exit"][bi, ti]
            feats, ces, oks = [], [], []
'''
new = '''            ex = out["exit"][bi, ti]
            if args.exact:
                ex = torch.full_like(ex, core.N_LOOPS - 1)   # with an exact cache every loop's state is a valid one
            feats, ces, oks = [], [], []
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''model.eval()
for p in model.parameters():
    p.requires_grad_(False)
'''
new = '''model.eval()
for p in model.parameters():
    p.requires_grad_(False)
FILL = core.load_fill_maps(args.fill_maps) if args.fill_maps else None
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''grid = [float(x) for x in args.grid.split(",")]
routers = [tf_eval.parse_router("full")] + [tf_eval.parse_router(str(q)) for q in grid]
frontier = {}
print("released gate:", flush=True)
frontier["released"] = tf_eval.evaluate(model, val, routers, tok.pad_token_id, args.budget_tok, log=lambda s: print(s, flush=True))
for v, g in gates.items():
    print(f"gate {v}:", flush=True)
    g16 = nn.Linear(dim, 1).cuda().to(torch.bfloat16)
    g16.load_state_dict(g.state_dict())
    frontier[v] = tf_eval.evaluate(model, val, routers[1:], tok.pad_token_id, args.budget_tok, gate=g16, log=lambda s: print(s, flush=True))
'''
new = '''routers = [tf_eval.parse_router(r) for r in args.routers.split(";")]
frontier = {}
print("released gate:", flush=True)
frontier["released"] = tf_eval.evaluate(model, val, routers, tok.pad_token_id, args.budget_tok, log=lambda s: print(s, flush=True), fill_maps=FILL)
for v, g in gates.items():
    print(f"gate {v}:", flush=True)
    g16 = nn.Linear(dim, 1).cuda().to(torch.bfloat16)
    g16.load_state_dict(g.state_dict())
    frontier[v] = tf_eval.evaluate(model, val, routers, tok.pad_token_id, args.budget_tok, gate=g16, log=lambda s: print(s, flush=True), fill_maps=FILL)
'''
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace('''Writes <OUT>/gates_<tag>/{<variant>.pt ({"weight" (1, 2048), "bias" (1,)}), fit.jsonl, frontier.json (teacher-forced
loops/agreement of every gate at a grid of single thresholds on the validation problems, next to the released gate)}.''',
              '''Writes <OUT>/gates_<tag>/{<variant>.pt ({"weight" (1, 2048), "bias" (1,)}), fit.jsonl, frontier.json (teacher-forced
loops/agreement of every gate under gate-ranked exit shares on the validation problems, next to the released gate; shares
instead of thresholds because the fitted gates' probabilities live on other scales than the released gate's)}.''')
p.write_text(s)
print("patched stage2.py")
