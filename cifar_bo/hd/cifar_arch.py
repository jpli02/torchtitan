"""11-knob CIFAR-10 objective: the 8 training knobs of cifar_hd.py plus 3 convolution-architecture knobs (user, 2026-11-25:
"cifar 10加conv的维度 跑一下").

The network is a CIFAR ResNet built here so its shape can vary: a 3x3 stem conv (stride 1, no max-pool, like the ResNet-18
of objective/cifar.py), then 4 stages of basic residual blocks with widths w, 2w, 4w, 8w and strides 1, 2, 2, 2, global
average pooling and a linear head. w = 64, 2 blocks per stage, kernel 3 is ResNet-18.

  x0..x7  as in cifar_hd.py (lr, momentum, weight_decay, log2_batch_size, label_smoothing, warmup_epochs, nesterov, mixup)
  x8  base_width        [16, 96]  channels of stage 1 = 8 * round(x8 / 8)
  x9  blocks_per_stage  [1, 3]    round(x9) residual blocks in every stage
  x10 kernel_size       [3, 5]    3 below 4, else 5, for every conv in the blocks (the stem stays 3x3)

Training runs under bf16 autocast (wider and deeper nets cost up to ~9x ResNet-18 in fp32); everything else (data,
augmentation, seeding, the per-step warm-up + cosine schedule, mixup, the -test_accuracy return) follows cifar_hd.py.
The objective is accuracy only, with no cost term. Registered at run time by run_hd.py as "cifar_arch".
"""
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

import torchvision
import torchvision.transforms as transforms

from objective.cifar import CifarObjective, _eval_accuracy, _eval_loss, _get_loaders, seed_worker
from objective.param import ObjectiveParam
from cifar_hd import KNOBS as TRAIN_KNOBS, decode as decode_train

# CIFAR-100 (user, 2026-11-26: "cifar100 跑一下吧"): same augmentation and loader settings as objective/cifar.py's CIFAR-10,
# with CIFAR-100's own normalisation; downloaded once (torchvision checks the archive's MD5) into its own directory.
CIFAR100_ROOT = os.environ.get("CIFAR100_ROOT", "/home/jli199/boptim_scratch/data/cifar100")
_c100 = None


def _get_loaders_c100(train_batch_size: int = 128, seed: int = None):
    global _c100
    if _c100 is None:
        norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), norm])
        test_tf = transforms.Compose([transforms.ToTensor(), norm])
        _c100 = (torchvision.datasets.CIFAR100(root=CIFAR100_ROOT, train=True, download=False, transform=train_tf),
                 torchvision.datasets.CIFAR100(root=CIFAR100_ROOT, train=False, download=False, transform=test_tf))
    trainset, testset = _c100
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return (
        torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True,
                                    generator=generator, worker_init_fn=seed_worker),
        torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True),
    )

KNOBS = TRAIN_KNOBS + [
    ("base_width", (16.0, 96.0)),
    ("blocks_per_stage", (1.0, 3.0)),
    ("kernel_size", (3.0, 5.0)),
]


def decode(params):
    s = decode_train(params[:8])
    width, blocks, kernel = (float(v) for v in params[8:11])
    s.update(base_width=int(8 * max(2, round(width / 8))), blocks_per_stage=int(min(3, max(1, round(blocks)))),
             kernel_size=3 if kernel < 4.0 else 5)
    return s


class Block(nn.Module):
    def __init__(self, cin, cout, stride, k):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(out)) + self.short(x))


class ResNetCifar(nn.Module):
    def __init__(self, width=64, blocks=2, k=3, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        layers, cin = [], width
        for i, stride in enumerate((1, 2, 2, 2)):
            cout = width * 2 ** i
            for j in range(blocks):
                layers.append(Block(cin, cout, stride if j == 0 else 1, k))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class CifarArchObjective(CifarObjective):
    dataset = "cifar10"

    def __init__(self, n_epochs: int = 10, seed: int = None, metric: str = "accuracy"):
        super().__init__(n_epochs=n_epochs, train_batch_size=128, seed=seed, metric=metric)
        self.params = [ObjectiveParam(name, bounds) for name, bounds in KNOBS]
        self.num_classes = 100 if self.dataset == "cifar100" else 10

    @property
    def description(self) -> str:
        name = "CIFAR-100" if self.dataset == "cifar100" else "CIFAR-10"
        return f"{name} ResNet test accuracy, 8 training knobs + 3 conv architecture knobs ({self.n_epochs} epochs, maximize)"

    @property
    def setup_info(self) -> dict:
        return {"dataset": self.dataset, "n_epochs": self.n_epochs, "knobs": [k for k, _ in KNOBS], "precision": "bf16 autocast",
                "architecture": "3x3 stem, 4 stages of basic blocks (w, 2w, 4w, 8w; strides 1,2,2,2), avg-pool, linear",
                "base_width": "8 * round(x / 8)", "blocks_per_stage": "round(x)", "kernel_size": "3 if x < 4 else 5"}

    def evaluate(self, params) -> float:
        s = decode(params)
        t0 = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.seed is not None:  # same seeding as CifarObjective.evaluate
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        if self.dataset == "cifar100":
            trainloader, testloader = _get_loaders_c100(s["batch_size"], self.seed)
        else:
            trainloader, testloader = _get_loaders(s["batch_size"], self.seed)
        net = ResNetCifar(s["base_width"], s["blocks_per_stage"], s["kernel_size"], num_classes=self.num_classes).to(device)
        n_params = sum(p.numel() for p in net.parameters())
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
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            test_accuracy = _eval_accuracy(net, testloader, device)
            test_loss = _eval_loss(net, testloader, nn.CrossEntropyLoss(), device)
        self._last_eval = {"test_accuracy": test_accuracy, "test_loss": test_loss, "training_loss": train_loss,
                           "params_m": n_params / 1e6, "eval_seconds": time.time() - t0, **s}
        if self.metric == "accuracy":
            return -test_accuracy
        return test_loss if self.metric == "test_loss" else train_loss


class CifarArch100Objective(CifarArchObjective):
    """The same 11 knobs on CIFAR-100 (100 classes); registered by run_hd.py as "cifar100_arch"."""
    dataset = "cifar100"
