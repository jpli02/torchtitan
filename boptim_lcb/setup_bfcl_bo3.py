"""Second step of installing the BFCL router objective into boptim-agent (idempotent; run after setup_bfcl_bo.py): lets a search
use another model directory and the calibrated cache fill.
  --ouro_bfcl_ckpt       model directory for every evaluation (the exit-lab directory for the options below)
  --ouro_bfcl_fill_maps  path of the fitted fill maps (decode_fill_maps): skipped loops get corrected cache entries
Also refreshes objective/ouro_bfcl_router.py from the worktree."""
import shutil

WT = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft"
BA = "/home/jli199/boptim-agent"

shutil.copy(f"{WT}/boptim_lcb/ouro_bfcl_router.py", f"{BA}/objective/ouro_bfcl_router.py")

p = f"{BA}/objective/__init__.py"
s = open(p).read()
if "ouro_bfcl_fill_maps" not in s:
    old = '''            work_dir=getattr(args, "ouro_bfcl_work_dir", None),
            seed=args.objective_seed,
        )
    elif name == "ouro_lcb_router":'''
    new = '''            work_dir=getattr(args, "ouro_bfcl_work_dir", None),
            seed=args.objective_seed,
            **({"ckpt": str(args.ouro_bfcl_ckpt)} if getattr(args, "ouro_bfcl_ckpt", None) else {}),
            extra_cfg=({"decode_fill_maps": str(args.ouro_bfcl_fill_maps)} if getattr(args, "ouro_bfcl_fill_maps", None) else None),
        )
    elif name == "ouro_lcb_router":'''
    assert s.count(old) == 1, s.count(old)
    open(p, "w").write(s.replace(old, new, 1))
    print("objective/__init__.py: ckpt and fill maps are passed to the BFCL objective")
else:
    print("objective/__init__.py already passes ckpt and fill maps")

p = f"{BA}/main.py"
s = open(p).read()
if "--ouro_bfcl_fill_maps" not in s:
    anchor = '    parser.add_argument("--ouro_bfcl_gpus", type=str, default=None,'
    assert s.count(anchor) == 1, s.count(anchor)
    block = ('    parser.add_argument("--ouro_bfcl_ckpt", type=str, default=None, help="[ouro_bfcl_router] model directory (default: the deployed early-exit directory)")\n'
             '    parser.add_argument("--ouro_bfcl_fill_maps", type=str, default=None, help="[ouro_bfcl_router] fitted cache-fill maps (exit-lab model): corrected entries for skipped loops")\n')
    open(p, "w").write(s.replace(anchor, block + anchor, 1))
    print("main.py: --ouro_bfcl_ckpt and --ouro_bfcl_fill_maps added")
else:
    print("main.py already has the flags")
