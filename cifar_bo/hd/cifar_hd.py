"""8-knob CIFAR-10 objective for the higher-dimensional optimizer comparison (user, 2026-11-23: "尝试更高维度搜索cifar实验").

The 3-knob objective (boptim-agent objective/cifar.py: lr, momentum, weight decay) is small enough that the GP-hedge baseline,
whose 10 points are all skopt grid points (n_initial = budget = 10), covers it; in 8 dimensions a 10-point grid cannot. This
keeps that objective's model, data, augmentation, seeding and metric, and adds five real training knobs:

  x0 lr               [0.001, 0.5]   peak learning rate (SGD)
  x1 momentum         [0, 0.99]
  x2 weight_decay     [0, 0.01]
  x3 log2_batch_size  [6, 9]         batch size = round(2 ** x3), 64..512
  x4 label_smoothing  [0, 0.3]
  x5 warmup_epochs    [0, 3]         linear warm-up from 0, then cosine decay to 0, both per step
  x6 nesterov         [0, 1]         on when >= 0.5 (and momentum > 0)
  x7 mixup_alpha      [0, 1]         mixup with Beta(a, a) per batch when a >= 0.05, off below

Returns -test_accuracy (internal minimisation, like CifarObjective, which it subclasses so boptim-agent's main.py treats it
the same way). Also defines RandomSearch, the memoryless uniform baseline (the trajectory checkout has none registered).
Registered at run time by run_hd.py; no boptim-agent file is changed.
"""
import math
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from objective.cifar import CifarObjective, _eval_accuracy, _eval_loss, _get_loaders, _resnet18
from objective.param import ObjectiveParam
from optim.base import BaseOptimizer

KNOBS = [
    ("lr", (1e-3, 5e-1)),
    ("momentum", (0.0, 0.99)),
    ("weight_decay", (0.0, 1e-2)),
    ("log2_batch_size", (6.0, 9.0)),
    ("label_smoothing", (0.0, 0.3)),
    ("warmup_epochs", (0.0, 3.0)),
    ("nesterov", (0.0, 1.0)),
    ("mixup_alpha", (0.0, 1.0)),
]


def decode(params):
    """The training settings a point stands for."""
    lr, momentum, weight_decay, log2_bs, label_smoothing, warmup_epochs, nesterov, mixup_alpha = (float(v) for v in params)
    return {
        "lr": lr, "momentum": momentum, "weight_decay": weight_decay,
        "batch_size": int(round(2.0 ** log2_bs)), "label_smoothing": label_smoothing, "warmup_epochs": warmup_epochs,
        "nesterov": bool(nesterov >= 0.5 and momentum > 0.0), "mixup_alpha": mixup_alpha if mixup_alpha >= 0.05 else 0.0,
    }


class CifarHDObjective(CifarObjective):
    def __init__(self, n_epochs: int = 10, seed: int = None, metric: str = "accuracy"):
        super().__init__(n_epochs=n_epochs, train_batch_size=128, seed=seed, metric=metric)
        self.params = [ObjectiveParam(name, bounds) for name, bounds in KNOBS]

    @property
    def description(self) -> str:
        return f"CIFAR-10 ResNet-18 test accuracy, 8 training knobs ({self.n_epochs} epochs, maximize)"

    @property
    def setup_info(self) -> dict:
        return {"n_epochs": self.n_epochs, "knobs": [k for k, _ in KNOBS],
                "schedule": "per-step linear warm-up over warmup_epochs, then cosine to 0",
                "batch_size": "round(2 ** log2_batch_size)", "nesterov": ">= 0.5", "mixup": "alpha >= 0.05"}

    def evaluate(self, params) -> float:
        s = decode(params)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.seed is not None:  # same seeding as CifarObjective.evaluate
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        trainloader, testloader = _get_loaders(s["batch_size"], self.seed)
        net = _resnet18(num_classes=10).to(device)
        if device == "cuda":
            net = torch.nn.DataParallel(net)
        criterion = nn.CrossEntropyLoss(label_smoothing=s["label_smoothing"])
        opt = optim.SGD(net.parameters(), lr=s["lr"], momentum=s["momentum"], weight_decay=s["weight_decay"],
                        nesterov=s["nesterov"])
        steps_per_epoch = len(trainloader)
        total = self.n_epochs * steps_per_epoch
        warm = min(int(round(s["warmup_epochs"] * steps_per_epoch)), total - 1)

        def lr_at(step):
            if step < warm:
                return s["lr"] * (step + 1) / warm
            return s["lr"] * 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

        beta = torch.distributions.Beta(s["mixup_alpha"], s["mixup_alpha"]) if s["mixup_alpha"] > 0 else None
        step, train_loss = 0, 0.0
        for _ in range(self.n_epochs):
            net.train()
            running = 0.0
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.zero_grad()
                if beta is not None:
                    lam = float(beta.sample())
                    perm = torch.randperm(inputs.size(0), device=inputs.device)
                    outputs = net(lam * inputs + (1.0 - lam) * inputs[perm])
                    loss = lam * criterion(outputs, targets) + (1.0 - lam) * criterion(outputs, targets[perm])
                else:
                    loss = criterion(net(inputs), targets)
                loss.backward()
                opt.step()
                running += loss.item()
                step += 1
            train_loss = running / steps_per_epoch
        test_accuracy = _eval_accuracy(net, testloader, device)
        test_loss = _eval_loss(net, testloader, nn.CrossEntropyLoss(), device)
        self._last_eval = {"test_accuracy": test_accuracy, "test_loss": test_loss, "training_loss": train_loss, **s}
        if self.metric == "accuracy":
            return -test_accuracy
        return test_loss if self.metric == "test_loss" else train_loss


class RandomSearch(BaseOptimizer):
    """Uniform random search inside the bounds; tell() is recorded but never used by ask()."""

    def __init__(self, bounds, seed=None):
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.best = SimpleNamespace(fun=float("inf"), x=None)

    def ask(self, n_points: int = 1) -> tuple:
        return [[float(self.rng.uniform(lo, hi)) for lo, hi in self.bounds] for _ in range(n_points)], ""

    def tell(self, x, y) -> tuple:
        # main.py prints result.fun / result.x (skopt's best, internal minimised scale) after the loop; never used by ask()
        for xi, yi in zip(x, y):
            if float(yi) < self.best.fun:
                self.best = SimpleNamespace(fun=float(yi), x=list(map(float, xi)))
        return self.best, ""

    @property
    def description(self) -> str:
        return f"RandomSearch(dim={len(self.bounds)}, seed={self.seed})"

    @property
    def setup_info(self) -> dict:
        return {"seed": self.seed, "memoryless": True}
