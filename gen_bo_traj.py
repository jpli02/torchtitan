#!/usr/bin/env python3
"""Generate raw BO trajectories WITH the optimizer's recorded reasoning.

The shipped generate_data.py hardcodes optim="gp_hedge" and its Observation
has no explanation field, so it cannot produce the kind of trajectory that
data/gpt_cifar_3d_seed0000.json is: an LLM optimizer's per-iteration
"Analysis: ... | Plan: ..." next to the point it chose. That reasoning is the
whole point of distilling into Qwen, so this driver runs the ask/tell loop
itself and writes the same raw schema as the CIFAR file.

  --optim chatgpt  -> reasoning trajectories (API cost: max_iter calls each)
  --optim gp_hedge -> point-only trajectories, free (train the JSON contract
                      and point selection; no analysis/plan)

Objectives come from boptim-agent's OBJECTIVE_REGISTRY. Synthetic ones (sumsq,
rosenbrock, fourier2d, ...) are free and fast; cifar is a 10-epoch ResNet per
evaluation and is NOT what you want for volume.

Usage (run from /home/jli199/boptim-agent):
  python gen_bo_traj.py --objective sumsq --optim chatgpt --model gpt-5-mini \
      --n 20 --seed_start 0 --max_iter 15 --out data/trajectories_gen
"""
import argparse
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

sys.path.insert(0, os.environ.get("BOPTIM_REPO", "/home/jli199/boptim-agent"))


def _load_keys(path=os.path.expanduser("~/.boptim_keys.env")):
    """Load API keys (OPENAI_API_KEY, QWEN_API_KEY, ...) from the user's env
    file into os.environ without echoing them; existing env vars win."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.replace("export ", "").strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


_load_keys()
from objective import build_objective  # noqa: E402
from optim import build_optimizer  # noqa: E402


def run_one(a, seed):
    args = SimpleNamespace(
        objective=a.objective, direction=a.direction, objective_seed=seed,
        optimizer_seed=seed, seed=seed, n_dims=a.n_dims, n_fourier=a.n_fourier,
        bounds=None, optim=a.optim, model=a.model, n_initial=a.n_initial,
        acq=a.acq, no_explanation=a.no_explanation, explanation=not a.no_explanation,
        max_iter=a.max_iter, batch_size=1,
        # cifar-only knobs, harmless for synthetic objectives
        epochs=a.epochs, cifar_metric="test_accuracy", metrics=None, train_batch_size=128,
    )
    obj = build_objective(args)
    opt = build_optimizer(args, bounds=obj.bounds, seed=seed)
    sign = -1.0 if a.direction == "maximize" else 1.0
    # LLMOptimizerMixin.ask() returns a fixed "<LLM> suggested N points" string
    # and drops the parsed "Analysis: ... | Plan: ..." explanation on the floor.
    # Wrap _parse_response to capture what the model actually said.
    captured = {}
    if hasattr(opt, "_parse_response"):
        _orig_parse = opt._parse_response

        def _capturing_parse(response, n_points):
            pts, expl = _orig_parse(response, n_points)
            captured["expl"] = expl
            return pts, expl

        opt._parse_response = _capturing_parse
    bounds = [[float(lo), float(hi)] for lo, hi in obj.bounds]
    pnames = getattr(obj, "param_names", None) or [f"x{i}" for i in range(len(bounds))]
    obs, best_v, best_p = [], None, None
    for it in range(a.max_iter):
        captured.pop("expl", None)
        pts, expl = opt.ask(1)
        expl = captured.get("expl") or expl
        if expl == "No explanations provided" or (isinstance(expl, str) and expl.endswith("suggested 1 points")):
            expl = ""
        p = [float(v) for v in pts[0]]
        v = float(obj.evaluate(p))
        opt.tell([p], [sign * v])
        better = best_v is None or (v > best_v if a.direction == "maximize" else v < best_v)
        if better:
            best_v, best_p = v, list(p)
        obs.append({"iteration": it, "point": [round(x, 6) for x in p], "value": round(v, 6),
                    "is_best": better, "best_value_so_far": round(best_v, 6),
                    "best_point_so_far": [round(x, 6) for x in best_p],
                    "explanation": (expl or "") if a.optim in ("chatgpt", "qwen", "claude") else ""})
        if a.optim in ("chatgpt", "qwen", "claude") and getattr(opt, "is_complete", False):
            break
    best_it = next((o["iteration"] for o in obs if o["is_best"]), 0)
    # last is_best wins (latest improvement)
    for o in obs:
        if o["is_best"]:
            best_it = o["iteration"]
    return {
        "trajectory_id": f"{a.objective}_{len(bounds)}d_{a.optim}_seed{seed:04d}",
        "objective": {"name": a.objective, "description": f"{a.objective} ({len(bounds)}D, {a.direction})",
                      "bounds": bounds, "param_names": pnames, "n_dims": len(bounds)},
        "optimizer_config": {"optim": a.optim, "model": a.model if a.optim in ("chatgpt", "qwen", "claude") else None,
                             "explanations": "disabled" if a.no_explanation else "enabled",
                             "n_initial_points": a.n_initial},
        "direction": a.direction, "seed": seed, "max_iterations": a.max_iter,
        "observations": obs,
        "final_result": {"best_point": best_p, "best_value": best_v, "n_iterations": len(obs),
                         "best_iteration": best_it, "stopped_early": len(obs) < a.max_iter},
        "timestamp": datetime.now().isoformat(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="sumsq")
    ap.add_argument("--direction", default="minimize", choices=["minimize", "maximize"])
    ap.add_argument("--optim", default="gp_hedge", choices=["gp_hedge", "chatgpt", "qwen", "claude", "random"])
    ap.add_argument("--model", default="gpt-5-mini",
                    help="LLM name for chatgpt/qwen/claude (qwen goes through QWEN_BASE_URL, e.g. qwen/qwen3.5-9b on OpenRouter)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--max_iter", type=int, default=15)
    ap.add_argument("--n_initial", type=int, default=10)
    ap.add_argument("--acq", default="gp_hedge")
    ap.add_argument("--n_dims", type=int, default=None)
    ap.add_argument("--n_fourier", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--no_explanation", action="store_true")
    ap.add_argument("--out", default="data/trajectories_gen")
    ap.add_argument("--qwen_base_url", default=None,
                    help="point the qwen optimizer at a local OpenAI-compatible server "
                         "(qwen_bo_server.py), e.g. http://127.0.0.1:8210/v1")
    a = ap.parse_args()
    if a.qwen_base_url:
        os.environ["QWEN_BACKEND"] = "api"
        os.environ["QWEN_BASE_URL"] = a.qwen_base_url
        os.environ["QWEN_API_KEY"] = "local"
        os.environ.setdefault("QWEN_MAX_TOKENS", "4096")
        os.environ.setdefault("QWEN_MAX_RETRIES", "2")
    os.makedirs(a.out, exist_ok=True)
    ok = 0
    for seed in range(a.seed_start, a.seed_start + a.n):
        try:
            t = run_one(a, seed)
        except Exception as e:  # keep going; report at the end
            print(f"  seed {seed}: FAILED {str(e)[:120]}")
            continue
        fp = os.path.join(a.out, f"{t['trajectory_id']}.json")
        json.dump(t, open(fp, "w"), indent=1)
        ok += 1
        print(f"  seed {seed}: {len(t['observations'])} obs, best {t['final_result']['best_value']:.4f} -> {fp}")
    print(f"generated {ok}/{a.n} trajectories ({a.objective}, {a.optim}) -> {a.out}")


if __name__ == "__main__":
    main()
