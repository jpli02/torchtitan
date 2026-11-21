"""Install the fast router-NAS objective (randrouter/ouro_tf_nas.py) in boptim-agent, next to ouro_tf_router: copy the file,
register the class and its factory branch in objective/__init__.py, add the --ouro_tfnas_* flags to main.py. Idempotent; the
objective file is copied every time. Files are replaced atomically (a run may be starting while this is called).

  /home/jli199/boptim-agent/.venv/bin/python randrouter/setup_tfnas_objective.py
"""
import os
import py_compile
import shutil
import tempfile

BO = "/home/jli199/boptim-agent"
SRC = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/randrouter/ouro_tf_nas.py"


def replace(path: str, text: str):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    py_compile.compile(tmp, doraise=True)
    shutil.copymode(path, tmp)
    os.replace(tmp, path)


tmp = f"{BO}/objective/.ouro_tf_nas.py.tmp"
shutil.copy(SRC, tmp)
py_compile.compile(tmp, doraise=True)
os.replace(tmp, f"{BO}/objective/ouro_tf_nas.py")

p = f"{BO}/objective/__init__.py"
s = open(p).read()
if "OuroTFNASObjective" not in s:
    a = "from .ouro_tf_router    import OuroTFRouterObjective"
    assert s.count(a) == 1, "import anchor not found in objective/__init__.py"
    s = s.replace(a, a + "\nfrom .ouro_tf_nas       import OuroTFNASObjective", 1)
    a = '    "ouro_tf_router": OuroTFRouterObjective,'
    assert s.count(a) == 1, "registry anchor not found in objective/__init__.py"
    s = s.replace(a, a + '\n    "ouro_tf_nas": OuroTFNASObjective,', 1)
    branch = '''    elif name == "ouro_tf_nas":
        obj = OuroTFNASObjective(
            feats=str(getattr(args, "ouro_tfnas_feats", "/home/jli199/boptim_scratch/randrouter/nas_feats_train.pt")),
            split=str(getattr(args, "ouro_tfnas_split", "val")),
            n_items=int(getattr(args, "ouro_tfnas_n", 99)),
            floor_pct=float(getattr(args, "ouro_tfnas_floor", 98.45)),
            penalty_per_pt=float(getattr(args, "ouro_tfnas_penalty", 3.5)),
            mem_gb=float(getattr(args, "ouro_tfnas_mem_gb", 0.0)),
            work_dir=getattr(args, "ouro_tfnas_work_dir", None),
            exact=int(getattr(args, "ouro_tfnas_exact", 0) or 0),
            per_loop=int(getattr(args, "ouro_tfnas_per_loop", 0) or 0),
            steps=int(getattr(args, "ouro_tfnas_steps", 300)),
            k=float(getattr(args, "ouro_tfnas_k", 50.0)),
            seed=args.objective_seed,
        )
'''
    anchor = '    elif name == "ouro_tf_router":'
    assert s.count(anchor) == 1, "factory anchor not found in objective/__init__.py"
    s = s.replace(anchor, branch + anchor, 1)
    replace(p, s)
    print("registered ouro_tf_nas in objective/__init__.py")
else:
    print("objective/__init__.py already has ouro_tf_nas")

p = f"{BO}/main.py"
s = open(p).read()
if "--ouro_tfnas_feats" not in s:
    block = '''    parser.add_argument("--ouro_tfnas_feats", type=str, default="/home/jli199/boptim_scratch/randrouter/nas_feats_train.pt", help="[ouro_tf_nas] cached states of the frozen model (randrouter/nas_feats.py)")
    parser.add_argument("--ouro_tfnas_split", type=str, default="val", help="[ouro_tf_nas] split of the teacher answers the router is scored on")
    parser.add_argument("--ouro_tfnas_n", type=int, default=99, help="[ouro_tf_nas] teacher answers per evaluation")
    parser.add_argument("--ouro_tfnas_floor", type=float, default=98.45, help="[ouro_tf_nas] agreement %% below which the penalty starts")
    parser.add_argument("--ouro_tfnas_penalty", type=float, default=3.5, help="[ouro_tf_nas] penalty per agreement point below the floor")
    parser.add_argument("--ouro_tfnas_mem_gb", type=float, default=0.0, help="[ouro_tf_nas] cap of the process's GPU memory")
    parser.add_argument("--ouro_tfnas_work_dir", type=str, default=None, help="[ouro_tf_nas] directory of evaluations.jsonl")
    parser.add_argument("--ouro_tfnas_exact", type=int, default=0, help="[ouro_tf_nas] 1 = score with an exact cache")
    parser.add_argument("--ouro_tfnas_per_loop", type=int, default=0, help="[ouro_tf_nas] 1 = three per-loop exit thresholds instead of one (8 dimensions)")
    parser.add_argument("--ouro_tfnas_steps", type=int, default=300, help="[ouro_tf_nas] Stage-II training steps of a candidate router")
    parser.add_argument("--ouro_tfnas_k", type=float, default=50.0, help="[ouro_tf_nas] sharpness k of the Stage-II label")
'''
    anchor = '    parser.add_argument("--ouro_tf_lora", type=str, default=None,'
    assert s.count(anchor) == 1, "anchor not found in main.py"
    s = s.replace(anchor, block + anchor, 1)
    replace(p, s)
    print("added the --ouro_tfnas_* flags to main.py")
else:
    print("main.py already has the --ouro_tfnas_* flags")
