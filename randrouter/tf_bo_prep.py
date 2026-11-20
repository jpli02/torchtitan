"""Prepare a router search on the teacher-forced proxy objective (ouro_tf_router) for one system (base / calibrated fill /
random-router LoRA / refitted gate): measure the released single-knob gate at several thresholds on the proxy (the baseline
the search has to beat, and its warm start, as in the real searches), write the boptim config and the resume state.

  CUDA_VISIBLE_DEVICES=9 python randrouter/tf_bo_prep.py --name fill24 [--fill_maps F] [--lora L] [--gate G] [--n 99] [--new 50]
        [--optim gp_hedge] [--model openai/gpt-6-luna] [--mem_gb 8.3]
then  cd ~/boptim-agent && CUDA_VISIBLE_DEVICES=9 ./.venv/bin/python -u run_experiment.py --config configs/ouro_tf_<name>.yaml --log-prefix tf_<name>
Everything of the run is under /home/jli199/boptim_scratch/randrouter/bo_tf/<name>/ (evaluations.jsonl, runs/ = resume state,
baseline.json = the single-knob table). The resume state lives in its own directory because the optimizer's state key does
not include the objective's own settings (two systems would otherwise share one state file).
"""
import argparse
import json
import os
import sys

BO = "/home/jli199/boptim-agent"
sys.path.insert(0, BO)

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--fill_maps", default=None)
ap.add_argument("--lora", default=None)
ap.add_argument("--gate", default=None)
ap.add_argument("--veto", default=None, help="harm probe: the loop-1 veto threshold becomes a fourth search variable")
ap.add_argument("--n", type=int, default=99)
ap.add_argument("--new", type=int, default=50, help="evaluations after the warm start")
ap.add_argument("--optim", default="gp_hedge")
ap.add_argument("--model", default=None, help="LLM of the chatgpt optimizer")
ap.add_argument("--knobs", default="1.0,0.7,0.5,0.3,0.25,0.245,0.2")
ap.add_argument("--floor", type=float, default=98.45)
ap.add_argument("--penalty", type=float, default=3.5)
ap.add_argument("--mem_gb", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

work = f"/home/jli199/boptim_scratch/randrouter/bo_tf/{args.name}"
os.makedirs(f"{work}/runs", exist_ok=True)
from objective.ouro_tf_router import OuroTFRouterObjective  # noqa: E402

obj = OuroTFRouterObjective(lora=args.lora, gate=args.gate, fill_maps=args.fill_maps, veto=args.veto, n_items=args.n, floor_pct=args.floor,
                            penalty_per_pt=args.penalty, mem_gb=args.mem_gb, work_dir=f"{work}/baseline")
print(obj.description, flush=True)
records, table = [], []
for q in [float(v) for v in args.knobs.split(",")]:
    x = [q, q, q] + ([3.0] if args.veto else [])        # 3.0 = the veto never fires
    v = obj.evaluate(x)
    rec = obj._eval_records[-1]
    records.append({"x": x, "fval": v})
    table.append({"q": q, "loops": rec["loops"], "agree_pct": rec["agree_pct"], "value": v, "exit_share": rec["exit_share"]})
json.dump({"system": obj.setup, "single_knob": table}, open(f"{work}/baseline.json", "w"), indent=1)

objective = {"type": "ouro_tf_router", "ouro_tf_n": args.n, "ouro_tf_floor": args.floor, "ouro_tf_penalty": args.penalty,
             "ouro_tf_work_dir": work}
for k, v in (("ouro_tf_fill_maps", args.fill_maps), ("ouro_tf_lora", args.lora), ("ouro_tf_gate", args.gate), ("ouro_tf_veto", args.veto)):
    if v:
        objective[k] = v
if args.mem_gb:
    objective["ouro_tf_mem_gb"] = args.mem_gb
max_iter = len(records) + args.new
lines = ["# Router search on the teacher-forced proxy of the LiveCodeBench objective (randrouter/ouro_tf_router.py), system: "
         + json.dumps(obj.setup), "# written by randrouter/tf_bo_prep.py; max_iter counts the warm-start records (the released single knob) plus the new evaluations",
         "replicates: 1", "", "wandb:", "  enabled: false", ""]
if args.model:
    lines += ["models:", f'  {args.optim}: "{args.model}"', ""]
lines += ["experiments:", f"  - optim: {args.optim}", "    objective:"]
lines += [f"      {k}: {v}" for k, v in objective.items()]
lines += ["    batch_size: 1", f"    max_iter: {max_iter}", "    resume: true", f"    resume_dir: {work}/runs",
          f"    optim_seed: {args.seed}", f"    objective_seed: {args.seed}"]
if args.optim == "gp_hedge":
    lines.append("    n_initial: 4")
cfg = f"{BO}/configs/ouro_tf_{args.name}.yaml"
open(cfg, "w").write("\n".join(lines) + "\n")

# the resume state run_experiment.py -> main.py will look for (same identity args; the objective's own keys are not part of the key)
import main as bo_main  # noqa: E402

argv = ["main.py", "--optim", args.optim, "--objective", "ouro_tf_router", "--batch_size", "1", "--max_iter", str(max_iter),
        "--optimizer-seed", str(args.seed), "--objective-seed", str(args.seed), "--resume", "True", "--resume_dir", f"{work}/runs"]
if args.model:
    argv += ["--model", args.model]
sys.argv = argv
state = bo_main._resume_state_path(bo_main.parse_args())
if state.exists() and bo_main._load_resume_records(state):
    print("state already has records; left as it is:", state)
else:
    json.dump({"records": records, "note": "warm start: the released single-knob gate on the proxy (randrouter/tf_bo_prep.py)"}, open(state, "w"), indent=1)
    print("state written:", state)
print(f"config {cfg}; single knob on the proxy:")
for t in table:
    print(f"  q={t['q']:<6} loops {t['loops']:.3f}  agreement {t['agree_pct']:.2f} %  value {t['value']:.3f}  exit shares {[round(100 * s) for s in t['exit_share']]}")
