"""Register OuroLCBRouterObjective in boptim-agent/objective/__init__.py (idempotent)."""
import re

p = "/home/jli199/boptim-agent/objective/__init__.py"
s = open(p).read()
if "OuroLCBRouterObjective" in s:
    print("already registered")
    raise SystemExit(0)
s = s.replace("from .ouro_router_nas   import OuroRouterNASObjective",
              "from .ouro_router_nas   import OuroRouterNASObjective\nfrom .ouro_lcb_router   import OuroLCBRouterObjective", 1)
s = s.replace('    "ouro_router_nas": OuroRouterNASObjective,',
              '    "ouro_router_nas": OuroRouterNASObjective,\n    "ouro_lcb_router": OuroLCBRouterObjective,', 1)
branch = '''    elif name == "ouro_lcb_router":
        obj = OuroLCBRouterObjective(
            mode=str(getattr(args, "ouro_lcb_mode", "steps")),
            n_items=int(getattr(args, "ouro_lcb_n", 120)),
            offset=int(getattr(args, "ouro_lcb_offset", 0)),
            max_new=int(getattr(args, "ouro_lcb_max_new", 1024)),
            baseline_pass1_pct=float(getattr(args, "ouro_lcb_baseline_acc", 22.5)),
            tol_pts=float(getattr(args, "ouro_lcb_tol", 1.0)),
            penalty_per_pt=float(getattr(args, "ouro_lcb_penalty", 2.0)),
            gpus=str(getattr(args, "ouro_lcb_gpus", None) or os.environ.get("LCB_BO_GPUS", "0")),
            work_dir=getattr(args, "ouro_lcb_work_dir", None),
            seed=args.objective_seed,
        )
'''
anchor = '    elif name in ("ouro", "ouro_humaneval", "ouro_mbpp", "ouro_router_nas"):'
assert anchor in s, "anchor not found"
s = s.replace(anchor, branch + anchor, 1)
if "\nimport os\n" not in s:
    s = s.replace("from .ouro_lcb_router   import OuroLCBRouterObjective", "import os\nfrom .ouro_lcb_router   import OuroLCBRouterObjective", 1)
open(p, "w").write(s)
print("registered ouro_lcb_router")
