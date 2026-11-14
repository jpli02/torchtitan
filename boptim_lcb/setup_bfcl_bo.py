"""Install the BFCL router objective into boptim-agent (idempotent): copy objective/ouro_bfcl_router.py, register it, add the
--ouro_bfcl_* flags to main.py, and build the search / held-out split of the 1,150 non-live items:
  bfcl_search.jsonl  = even positions within each category (575), the set the optimiser sees
  bfcl_heldout.jsonl = odd positions within each category (575), never seen by the search
(both in the grader's stage directory /srv/scratch/jli199/bfcl_stage; the live categories, bfcl_live_all.jsonl, are a second
held-out set)."""
import collections
import json
import shutil

WT = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft"
BA = "/home/jli199/boptim-agent"
STAGE = "/srv/scratch/jli199/bfcl_stage"

shutil.copy(f"{WT}/boptim_lcb/ouro_bfcl_router.py", f"{BA}/objective/ouro_bfcl_router.py")

p = f"{BA}/objective/__init__.py"
s = open(p).read()
if "OuroBFCLRouterObjective" not in s:
    s = s.replace("from .ouro_lcb_router   import OuroLCBRouterObjective", "from .ouro_lcb_router   import OuroLCBRouterObjective\nfrom .ouro_bfcl_router  import OuroBFCLRouterObjective", 1)
    s = s.replace('    "ouro_lcb_router": OuroLCBRouterObjective,', '    "ouro_lcb_router": OuroLCBRouterObjective,\n    "ouro_bfcl_router": OuroBFCLRouterObjective,', 1)
    branch = '''    elif name == "ouro_bfcl_router":
        _b = getattr(args, "ouro_bfcl_baseline_acc", None)
        obj = OuroBFCLRouterObjective(
            mode=str(getattr(args, "ouro_bfcl_mode", "steps")),
            data_name=str(getattr(args, "ouro_bfcl_data", "bfcl_search.jsonl")),
            n_items=int(getattr(args, "ouro_bfcl_n", 575)),
            baseline_acc_pct=(float(_b) if _b is not None else None),
            tol_pts=float(getattr(args, "ouro_bfcl_tol", 1.7)),
            penalty_per_pt=float(getattr(args, "ouro_bfcl_penalty", 5.0)),
            gpus=str(getattr(args, "ouro_bfcl_gpus", None) or os.environ.get("BFCL_BO_GPUS", "3")),
            shards=int(getattr(args, "ouro_bfcl_shards", 2)),
            work_dir=getattr(args, "ouro_bfcl_work_dir", None),
            seed=args.objective_seed,
        )
'''
    anchor = '    elif name == "ouro_lcb_router":'
    assert s.count(anchor) == 1 and "OuroBFCLRouterObjective" in s
    s = s.replace(anchor, branch + anchor, 1)
    open(p, "w").write(s)
    print("registered ouro_bfcl_router")
else:
    print("ouro_bfcl_router already registered")

p = f"{BA}/main.py"
s = open(p).read()
if "--ouro_bfcl_mode" not in s:
    block = '''    parser.add_argument("--ouro_bfcl_mode", type=str, default="steps", help="[ouro_bfcl_router] 'steps' = per-loop thresholds (q1,q2,q3); 'scalar' = one gate threshold")
    parser.add_argument("--ouro_bfcl_data", type=str, default="bfcl_search.jsonl", help="[ouro_bfcl_router] search-set file in the grader's stage directory")
    parser.add_argument("--ouro_bfcl_n", type=int, default=575, help="[ouro_bfcl_router] items per evaluation")
    parser.add_argument("--ouro_bfcl_baseline_acc", type=float, default=None, help="[ouro_bfcl_router] full-depth accuracy %% on the search set")
    parser.add_argument("--ouro_bfcl_tol", type=float, default=1.7, help="[ouro_bfcl_router] accuracy points allowed below baseline before the penalty")
    parser.add_argument("--ouro_bfcl_penalty", type=float, default=5.0, help="[ouro_bfcl_router] relative-compute points per accuracy point below the floor")
    parser.add_argument("--ouro_bfcl_shards", type=int, default=2, help="[ouro_bfcl_router] concurrent batch-1 processes per card")
    parser.add_argument("--ouro_bfcl_work_dir", type=str, default=None, help="[ouro_bfcl_router] directory for per-evaluation outputs")
    parser.add_argument("--ouro_bfcl_gpus", type=str, default=None, help="[ouro_bfcl_router] comma-separated GPU ids (default env BFCL_BO_GPUS or 3)")
'''
    anchor = '    parser.add_argument(\n        "--ouro_lcb_mode", type=str, default="steps",'
    assert s.count(anchor) == 1
    open(p, "w").write(s.replace(anchor, block + anchor, 1))
    print("main.py: --ouro_bfcl_* flags added")
else:
    print("main.py already has the flags")

rows = [l for l in open(f"{STAGE}/bfcl_all.jsonl") if l.strip()]
pos = collections.Counter()
search, held = [], []
for l in rows:
    c = json.loads(l)["category"]
    (search if pos[c] % 2 == 0 else held).append(l)
    pos[c] += 1
for name, part in (("bfcl_search.jsonl", search), ("bfcl_heldout.jsonl", held)):
    open(f"{STAGE}/{name}", "w").write("".join(l if l.endswith("\n") else l + "\n" for l in part))
    print(name, len(part), dict(collections.Counter(json.loads(l)["category"] for l in part)))
