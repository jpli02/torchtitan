"""Run boptim-agent's own main.py on the 8-knob CIFAR objective, without changing any boptim-agent file: put the trajectory
checkout first on sys.path, register objective "cifar_hd" and optimizer "random" in its registries, wrap the two factories,
then hand over to main.main() with the command line unchanged.

  cd <cifar-save-trajectory checkout> && python <worktree>/cifar_bo/hd/run_hd.py --objective cifar_hd --optim chatgpt \
      --model gpt-6-luna --max_iter 10 --optimizer-seed 42 --objective-seed 42 --n_epochs 10 --metrics validation_accuracy \
      --trajectory-dir <dir>
"""
import os
import sys

BOPTIM = os.environ.get("BOPTIM_REPO", "/home/jli199/boptim-agent/.claude/worktrees/cifar-save-trajectory")
sys.path.insert(0, BOPTIM)

import objective  # noqa: E402
import optim  # noqa: E402
from cifar_hd import CifarHDObjective, RandomSearch  # noqa: E402

objective.OBJECTIVE_REGISTRY["cifar_hd"] = CifarHDObjective
optim.OPTIMIZER_REGISTRY["random"] = RandomSearch
_build_objective, _build_optimizer = objective.build_objective, optim.build_optimizer


def build_objective(args, **kwargs):
    if args.objective == "cifar_hd":
        metric = getattr(args, "metrics", None) or getattr(args, "cifar_metric", "accuracy")
        return CifarHDObjective(n_epochs=getattr(args, "n_epochs", 10), seed=args.objective_seed, metric=str(metric))
    return _build_objective(args, **kwargs)


def build_optimizer(args, bounds, seed=None):
    if args.optim == "random":
        return RandomSearch(bounds, seed=seed)
    return _build_optimizer(args, bounds, seed=seed)


objective.build_objective, optim.build_optimizer = build_objective, build_optimizer

import main as boptim_main  # noqa: E402  (binds the wrapped factories)

assert boptim_main.build_objective is build_objective and boptim_main.build_optimizer is build_optimizer
boptim_main.main()
