"""
optim/random_search.py
----------------------
Uniform random search over the objective bounds.

This is the baseline every optimizer comparison needs and none of the others
provide. Without it, "gp_hedge scored X and the LLM scored Y" is unfalsifiable:
on low-dimensional spaces with small budgets random search is frequently
competitive with -- and sometimes better than -- model-based BO, so a method
that merely beats the other method may still not beat chance.

It is also the correct floor for the specific claim being tested here, that an
LLM agent can do NAS through BO. An LLM that cannot beat uniform sampling of the
architecture space is not doing search, whatever else it is doing.

Deliberately memoryless: tell() records history for reporting but never
influences ask(). That is the point of the baseline.
"""

import numpy as np

from .base import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    """Sample each coordinate uniformly at random inside its bounds."""

    def __init__(self, bounds, seed: int | None = None):
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.X: list[list[float]] = []
        self.y: list[float] = []

    def ask(self, n_points: int = 1) -> tuple:
        pts = [[float(self.rng.uniform(lo, hi)) for lo, hi in self.bounds]
               for _ in range(n_points)]
        return pts, ""

    def tell(self, x, y) -> tuple:
        # Recorded for reporting only -- never consulted by ask().
        self.X.extend([list(map(float, xi)) for xi in x])
        self.y.extend(float(v) for v in y)
        return None, ""

    @property
    def description(self) -> str:
        return f"RandomSearch(dim={len(self.bounds)}, seed={self.seed})"

    @property
    def setup_info(self) -> dict:
        return {"bounds": self.bounds, "seed": self.seed, "memoryless": True}
