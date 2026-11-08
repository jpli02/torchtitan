"""Add the --ouro_lcb_* flags (forwarded by run_experiment.py from the yaml objective dict) to boptim-agent/main.py."""
p = "/home/jli199/boptim-agent/main.py"
s = open(p).read()
if "--ouro_lcb_mode" in s:
    print("already patched")
    raise SystemExit(0)
block = '''    parser.add_argument(
        "--ouro_lcb_mode", type=str, default="steps",
        help="[ouro_lcb_router] 'steps' = per-step exit thresholds (q1,q2,q3); 'scalar' = one gate threshold"
    )
    parser.add_argument("--ouro_lcb_n", type=int, default=120, help="[ouro_lcb_router] LiveCodeBench problems per evaluation")
    parser.add_argument("--ouro_lcb_offset", type=int, default=0, help="[ouro_lcb_router] first problem index (held-out slices start at 120)")
    parser.add_argument("--ouro_lcb_max_new", type=int, default=1024, help="[ouro_lcb_router] generation budget per problem")
    parser.add_argument("--ouro_lcb_baseline_acc", type=float, default=22.5, help="[ouro_lcb_router] full-depth pass@1 %% on the slice")
    parser.add_argument("--ouro_lcb_tol", type=float, default=1.0, help="[ouro_lcb_router] accuracy points allowed below baseline before the penalty")
    parser.add_argument("--ouro_lcb_penalty", type=float, default=2.0, help="[ouro_lcb_router] penalty per accuracy point below the floor (in tok/s units)")
    parser.add_argument("--ouro_lcb_work_dir", type=str, default=None, help="[ouro_lcb_router] directory for per-evaluation generations/grades")
    parser.add_argument("--ouro_lcb_gpus", type=str, default=None, help="[ouro_lcb_router] comma-separated GPU ids (default env LCB_BO_GPUS or 0)")
'''
anchor = '    parser.add_argument(\n        "--bounds", type=float, nargs="+", default=None,'
assert anchor in s, "anchor not found"
s = s.replace(anchor, block + anchor, 1)
open(p, "w").write(s)
print("patched main.py")
