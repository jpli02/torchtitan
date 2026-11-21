"""Install the gate-NAS objective on LiveCodeBench (boptim_lcb/ouro_lcb_gate_nas.py) in boptim-agent: copy it and its parent
(ouro_lcb_router.py, via setup_lcb_options.py), register "ouro_lcb_gate_nas" in objective/__init__.py and add its --ouro_gnas_* flags
to main.py; the shared LiveCodeBench flags (--ouro_lcb_n, _baseline_acc, _tol, _penalty, _work_dir, _gpus, _data, _ckpt) are reused.
Idempotent; files are replaced atomically.

  /home/jli199/boptim-agent/.venv/bin/python boptim_lcb/setup_gate_nas.py
"""
import os
import py_compile
import shutil
import subprocess
import tempfile

BO = "/home/jli199/boptim-agent"
WT = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft"


def replace(path: str, text: str):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    py_compile.compile(tmp, doraise=True)
    if os.path.exists(path):
        shutil.copymode(path, tmp)
    os.replace(tmp, path)


subprocess.run(["/home/jli199/boptim-agent/.venv/bin/python", f"{WT}/boptim_lcb/setup_lcb_options.py"], check=True, capture_output=True)
replace(f"{BO}/objective/ouro_lcb_gate_nas.py", open(f"{WT}/boptim_lcb/ouro_lcb_gate_nas.py").read())

p = f"{BO}/objective/__init__.py"
s = open(p).read()
if "OuroLCBGateNASObjective" not in s:
    a = "from .ouro_lcb_router   import OuroLCBRouterObjective"
    assert s.count(a) == 1, "import anchor not found"
    s = s.replace(a, a + "\nfrom .ouro_lcb_gate_nas  import OuroLCBGateNASObjective", 1)
    a = '    "ouro_lcb_router": OuroLCBRouterObjective,'
    assert s.count(a) == 1, "registry anchor not found"
    s = s.replace(a, a + '\n    "ouro_lcb_gate_nas": OuroLCBGateNASObjective,', 1)
    branch = '''    elif name == "ouro_lcb_gate_nas":
        obj = OuroLCBGateNASObjective(
            n_items=int(getattr(args, "ouro_lcb_n", 342)),
            offset=int(getattr(args, "ouro_lcb_offset", 0)),
            max_new=int(getattr(args, "ouro_lcb_max_new", 1024)),
            baseline_pass1_pct=float(getattr(args, "ouro_lcb_baseline_acc", 20.5)),
            tol_pts=float(getattr(args, "ouro_lcb_tol", 2.2)),
            penalty_per_pt=float(getattr(args, "ouro_lcb_penalty", 2.0)),
            gpus=str(getattr(args, "ouro_lcb_gpus", None) or os.environ.get("LCB_BO_GPUS", "0")),
            work_dir=getattr(args, "ouro_lcb_work_dir", None),
            seed=args.objective_seed,
            **({"data": args.ouro_lcb_data} if getattr(args, "ouro_lcb_data", None) else {}),
            **({"ckpt": args.ouro_lcb_ckpt} if getattr(args, "ouro_lcb_ckpt", None) else {}),
            feats=str(getattr(args, "ouro_gnas_feats", None) or "/home/jli199/boptim_scratch/randrouter/nas_feats_train.pt"),
            shortcut=int(getattr(args, "ouro_gnas_shortcut", 1)),
            proxy=int(getattr(args, "ouro_gnas_proxy", 1)),
        )
'''
    anchor = '    elif name == "ouro_lcb_router":'
    assert s.count(anchor) == 1, "factory anchor not found"
    s = s.replace(anchor, branch + anchor, 1)
    replace(p, s)
    print("registered ouro_lcb_gate_nas")
else:
    print("ouro_lcb_gate_nas already registered")

p = f"{BO}/main.py"
s = open(p).read()
if "--ouro_gnas_feats" not in s:
    a = '    parser.add_argument("--ouro_lcb_gpus", type=str, default=None,'
    assert s.count(a) == 1, "main.py anchor not found"
    block = ('    parser.add_argument("--ouro_gnas_feats", type=str, default=None, help="[ouro_lcb_gate_nas] cached states for the router training (randrouter/nas_feats.py)")\n'
             '    parser.add_argument("--ouro_gnas_shortcut", type=int, default=1, help="[ouro_lcb_gate_nas] score degenerate routers (all exit after loop 1 / none exits) from measured runs")\n'
             '    parser.add_argument("--ouro_gnas_proxy", type=int, default=1, help="[ouro_lcb_gate_nas] record the teacher-forced proxy of each trained router")\n')
    s = s.replace(a, block + a, 1)
    replace(p, s)
    print("added --ouro_gnas_* flags")
else:
    print("main.py already has the --ouro_gnas_* flags")
