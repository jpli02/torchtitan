"""Offline check of the second (exit-share) search's plumbing: the runner's command line parses, builds the objective in shares
mode with the right box, maps shares to thresholds and back, and resolves to the state file bfcl_bo_run2.py writes. No evaluation."""
import sys

sys.path.insert(0, "/home/jli199/boptim-agent")
import main as bo_main  # noqa: E402
from objective import build_objective  # noqa: E402

WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
B = ["0", "0.0", "0.25", "1", "0.7", "1.0", "2", "0.5", "1.0"]
runner = ["main.py", "--optim", "gp_hedge", "--objective", "ouro_bfcl_router", "--batch_size", "2", "--max_iter", "19",
          "--optimizer-seed", "12", "--objective-seed", "12", "--wandb", "--wandb-project", "boptim-agent",
          "--wandb-group", "gp_hedge_ouro_bfcl_router", "--wandb-run-name", "gp_hedge_ouro_bfcl_router_r1",
          "--n_initial", "18", "--resume", "True", "--bounds"] + B + [
          "--ouro_bfcl_mode", "shares", "--ouro_bfcl_data", "bfcl_search.jsonl", "--ouro_bfcl_n", "575",
          "--ouro_bfcl_baseline_acc", "82.7826", "--ouro_bfcl_tol", "1.7", "--ouro_bfcl_penalty", "15.0", "--ouro_bfcl_shards", "4",
          "--ouro_bfcl_work_dir", WORK]
driver = ["main.py", "--optim", "gp_hedge", "--objective", "ouro_bfcl_router", "--batch_size", "2", "--max_iter", "19",
          "--optimizer-seed", "12", "--objective-seed", "12", "--n_initial", "18", "--resume", "True",
          "--ouro_bfcl_mode", "shares", "--ouro_bfcl_work_dir", WORK, "--bounds"] + B
sys.argv = runner
a = bo_main.parse_args()
obj = build_objective(a)
print("objective:", type(obj).__name__, "| mode", obj.mode, "| floor", obj.acc_floor, "| penalty", obj.penalty_per_pt, "| box", [(p.name, p.bounds) for p in obj.params])
for f in ([0.0, 0.85, 1.0], [0.126, 0.892, 1.0], [0.05, 0.9, 0.7], [0.25, 0.7, 0.5], [0.0, 1.0, 1.0]):
    q = obj._thresholds(f)
    back = obj.thresholds_to_shares(q)
    print(f"  shares {f} -> thresholds {q[:3]} -> shares {[round(v, 3) for v in back]}  loops {obj.loops_of_shares(back):.3f}")
p_runner = bo_main._resume_state_path(a)
sys.argv = driver
p_driver = bo_main._resume_state_path(bo_main.parse_args())
print("state (runner):", p_runner.name, "| state (driver):", p_driver.name, "| same:", p_runner == p_driver)
