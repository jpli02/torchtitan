"""The proxy objectives' value in the units of the real LiveCodeBench router objective.

    real objective   value = -6.25 * speed-up + 2.0 * max(0, 18.3 - pass@1 %)                       (boptim_lcb/ouro_lcb_router.py)
    proxy            value = -6.25 * 4 / loops + 2.0 * max(0, 18.3 - predicted pass@1 %)
    predicted pass@1 = c - alpha * own - beta * cache                                              (randrouter/calib_pass1.py)

own = % of answer tokens whose exit readout is wrong even with an exact cache (the token left too early), cache = % that are
right with an exact cache and wrong with the decoder's cache. The coefficients are fitted on every router that was run for real
on the 342 test problems; they are read from calib_pass1.json, so a later calibration with more real runs changes the
objective's version: the file's coefficients are written into every evaluation record.
"""
from __future__ import annotations

import json

CALIB = "/home/jli199/boptim_scratch/randrouter/calib_pass1.json"
FLOOR_PASS1 = 18.3          # full depth 20.5 % minus the tolerance 2.2 of the real objective
PENALTY_PER_PASS1_PT = 2.0


def load_calib(path: str = CALIB) -> dict:
    m = json.load(open(path))["model"]
    return {"c": float(m["c"]), "alpha": float(m["alpha"]), "beta": float(m["beta"]), "path": path}


def pass1_hat(own: float, cache: float, m: dict) -> float:
    return min(m["c"], max(0.0, m["c"] - m["alpha"] * max(0.0, own) - m["beta"] * max(0.0, cache)))


def value(loops: float, own: float, cache: float, m: dict, floor: float = FLOOR_PASS1, penalty: float = PENALTY_PER_PASS1_PT):
    """Returns (value, predicted pass@1)."""
    p = pass1_hat(own, cache, m)
    return -6.25 * 4.0 / loops + penalty * max(0.0, floor - p), p
