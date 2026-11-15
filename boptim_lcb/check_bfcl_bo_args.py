"""Offline check of the BO phase's plumbing (no evaluation, nothing written): the command line that run_experiment.py builds from
configs/ouro_bfcl_router_gp.yaml must parse in main.py, build the BFCL objective with the measured floor, and resolve to the same
resume-state file that bfcl_bo_run.py writes its warm start to."""
import sys

sys.path.insert(0, "/home/jli199/boptim-agent")
import main as bo_main  # noqa: E402
from objective import build_objective  # noqa: E402

WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
runner = ["main.py", "--optim", "gp_hedge", "--objective", "ouro_bfcl_router", "--batch_size", "2", "--max_iter", "13",
          "--optimizer-seed", "11", "--objective-seed", "11", "--wandb", "--wandb-project", "boptim-agent",
          "--wandb-group", "gp_hedge_ouro_bfcl_router", "--wandb-run-name", "gp_hedge_ouro_bfcl_router_r1",
          "--n_initial", "4", "--resume", "True", "--ouro_bfcl_mode", "steps", "--ouro_bfcl_data", "bfcl_search.jsonl", "--ouro_bfcl_n", "575",
          "--ouro_bfcl_baseline_acc", "82.7826", "--ouro_bfcl_tol", "1.7", "--ouro_bfcl_penalty", "5.0", "--ouro_bfcl_shards", "4",
          "--ouro_bfcl_work_dir", WORK]
driver = ["main.py", "--optim", "gp_hedge", "--objective", "ouro_bfcl_router", "--batch_size", "2", "--max_iter", "13",
          "--optimizer-seed", "11", "--objective-seed", "11", "--n_initial", "4", "--resume", "True"]
sys.argv = runner
a = bo_main.parse_args()
obj = build_objective(a)
print("objective:", type(obj).__name__, "| floor", obj.acc_floor, "| shards", obj.shards, "| gpus", obj.gpus, "| n", obj.n_items, "| data", obj.data)
print("bounds for 4 shards:", obj._bounds(4))
p_runner = bo_main._resume_state_path(a)
sys.argv = driver
p_driver = bo_main._resume_state_path(bo_main.parse_args())
print("state (runner):", p_runner)
print("state (driver):", p_driver, "| same:", p_runner == p_driver, "| exists:", p_driver.exists())
