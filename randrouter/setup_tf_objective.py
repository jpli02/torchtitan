"""Install the teacher-forced router objective (randrouter/ouro_tf_router.py) in boptim-agent: copy the file, register the
class and its factory branch in objective/__init__.py, add the --ouro_tf_* flags to main.py (run_experiment.py forwards the
yaml objective keys as flags). Idempotent; the objective file itself is copied every time.

  /home/jli199/boptim-agent/.venv/bin/python randrouter/setup_tf_objective.py
"""
import py_compile
import shutil

BO = "/home/jli199/boptim-agent"
SRC = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/randrouter/ouro_tf_router.py"

shutil.copy(SRC, f"{BO}/objective/ouro_tf_router.py")
py_compile.compile(f"{BO}/objective/ouro_tf_router.py", doraise=True)

p = f"{BO}/objective/__init__.py"
s = open(p).read()
if "OuroTFRouterObjective" not in s:
    a = "from .ouro_lcb_router   import OuroLCBRouterObjective"
    assert a in s
    s = s.replace(a, a + "\nfrom .ouro_tf_router    import OuroTFRouterObjective", 1)
    a = '    "ouro_lcb_router": OuroLCBRouterObjective,'
    assert a in s
    s = s.replace(a, a + '\n    "ouro_tf_router": OuroTFRouterObjective,', 1)
    branch = '''    elif name == "ouro_tf_router":
        obj = OuroTFRouterObjective(
            lora=getattr(args, "ouro_tf_lora", None),
            gate=getattr(args, "ouro_tf_gate", None),
            fill_maps=getattr(args, "ouro_tf_fill_maps", None),
            split=str(getattr(args, "ouro_tf_split", "val")),
            n_items=int(getattr(args, "ouro_tf_n", 99)),
            floor_pct=float(getattr(args, "ouro_tf_floor", 98.45)),
            penalty_per_pt=float(getattr(args, "ouro_tf_penalty", 3.5)),
            mem_gb=float(getattr(args, "ouro_tf_mem_gb", 0.0)),
            work_dir=getattr(args, "ouro_tf_work_dir", None),
            seed=args.objective_seed,
        )
'''
    anchor = '    elif name == "ouro_lcb_router":'
    assert anchor in s, "anchor not found in objective/__init__.py"
    s = s.replace(anchor, branch + anchor, 1)
    open(p, "w").write(s)
    py_compile.compile(p, doraise=True)
    print("registered ouro_tf_router in objective/__init__.py")
else:
    print("objective/__init__.py already has ouro_tf_router")
s = open(p).read()
if "ouro_tf_veto" not in s:
    a = '            fill_maps=getattr(args, "ouro_tf_fill_maps", None),\n'
    assert s.count(a) == 1
    s = s.replace(a, a + '            veto=getattr(args, "ouro_tf_veto", None),\n', 1)
    open(p, "w").write(s)
    py_compile.compile(p, doraise=True)
    print("added veto to the ouro_tf_router factory branch")
s = open(p).read()
if "ouro_tf_exact" not in s:
    a = '            veto=getattr(args, "ouro_tf_veto", None),\n'
    assert s.count(a) == 1
    s = s.replace(a, a + '            exact=int(getattr(args, "ouro_tf_exact", 0) or 0),\n', 1)
    open(p, "w").write(s)
    py_compile.compile(p, doraise=True)
    print("added exact to the ouro_tf_router factory branch")

p = f"{BO}/main.py"
s = open(p).read()
if "--ouro_tf_lora" not in s:
    block = '''    parser.add_argument("--ouro_tf_lora", type=str, default=None, help="[ouro_tf_router] LoRA state of the random-router training")
    parser.add_argument("--ouro_tf_gate", type=str, default=None, help="[ouro_tf_router] refitted exit gate")
    parser.add_argument("--ouro_tf_fill_maps", type=str, default=None, help="[ouro_tf_router] calibrated cache fill")
    parser.add_argument("--ouro_tf_split", type=str, default="val", help="[ouro_tf_router] split of the teacher answers")
    parser.add_argument("--ouro_tf_n", type=int, default=99, help="[ouro_tf_router] teacher answers per evaluation")
    parser.add_argument("--ouro_tf_floor", type=float, default=98.45, help="[ouro_tf_router] agreement %% below which the penalty starts")
    parser.add_argument("--ouro_tf_penalty", type=float, default=3.5, help="[ouro_tf_router] penalty per agreement point below the floor")
    parser.add_argument("--ouro_tf_mem_gb", type=float, default=0.0, help="[ouro_tf_router] cap of the process's GPU memory")
    parser.add_argument("--ouro_tf_work_dir", type=str, default=None, help="[ouro_tf_router] directory of evaluations.jsonl")
'''
    anchor = '    parser.add_argument(\n        "--bounds", type=float, nargs="+", default=None,'
    assert anchor in s, "anchor not found in main.py"
    s = s.replace(anchor, block + anchor, 1)
    open(p, "w").write(s)
    py_compile.compile(p, doraise=True)
    print("added the --ouro_tf_* flags to main.py")
else:
    print("main.py already has the --ouro_tf_* flags")
s = open(p).read()
if "--ouro_tf_veto" not in s:
    a = '    parser.add_argument("--ouro_tf_fill_maps", type=str, default=None, help="[ouro_tf_router] calibrated cache fill")\n'
    assert s.count(a) == 1
    s = s.replace(a, a + '    parser.add_argument("--ouro_tf_veto", type=str, default=None, help="[ouro_tf_router] harm probe: adds the loop-1 veto threshold as a fourth variable")\n', 1)
    open(p, "w").write(s)
    py_compile.compile(p, doraise=True)
    print("added the --ouro_tf_veto flag to main.py")
s = open(p).read()
if "--ouro_tf_exact" not in s:
    a = '    parser.add_argument("--ouro_tf_fill_maps", type=str, default=None, help="[ouro_tf_router] calibrated cache fill")\n'
    assert s.count(a) == 1
    s = s.replace(a, a + '    parser.add_argument("--ouro_tf_exact", type=int, default=0, help="[ouro_tf_router] 1 = exact cache (upper bound of a cache repair)")\n', 1)
    open(p, "w").write(s)
    py_compile.compile(p, doraise=True)
    print("added the --ouro_tf_exact flag to main.py")
