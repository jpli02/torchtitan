"""Smoke test and reference points of the fast router-NAS objective (ouro_tf_nas.py), outside the optimizer loop.

Evaluates a few hand-picked configurations (the linear router at the paper's gamma and at the gamma = 1 / threshold = 0.8
point that boptim-agent's notes call tuned for the linear gate, small and large MLP routers) and then N uniformly random points
of the search box, and prints how the values are distributed. The random points are a first look at the landscape: how rare
is a configuration that beats the released gate's single knob (-10.90 on this proxy)?

  CUDA_VISIBLE_DEVICES=0 python randrouter/nas_probe.py --mem_gb 8.5 --random 20 --out /home/jli199/boptim_scratch/randrouter/nas_probe
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/jli199/boptim-agent")

REFERENCE = [   # (layers, hidden, act, norm) as axis indices, gamma, threshold
    ([0, 1, 1, 0], 0.005, 0.3), ([0, 1, 1, 0], 0.005, 0.5), ([0, 1, 1, 0], 0.005, 0.7),
    ([0, 1, 1, 0], 0.05, 0.3), ([0, 1, 1, 0], 1.0, 0.8), ([0, 1, 1, 0], 0.2, 0.5),
    ([1, 1, 1, 0], 0.005, 0.3), ([2, 2, 0, 1], 0.05, 0.5), ([0, 1, 1, 1], 0.005, 0.3),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem_gb", type=float, default=8.5)
    ap.add_argument("--random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exact", type=int, default=0)
    ap.add_argument("--per_loop", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--no_reference", action="store_true")
    ap.add_argument("--out", default="/home/jli199/boptim_scratch/randrouter/nas_probe")
    a = ap.parse_args()
    from objective.ouro_tf_nas import OuroTFNASObjective

    os.makedirs(a.out, exist_ok=True)
    obj = OuroTFNASObjective(mem_gb=a.mem_gb, work_dir=a.out, exact=a.exact, per_loop=a.per_loop, steps=a.steps, seed=0)
    print(obj.description, flush=True)
    bounds = [p.bounds for p in obj.params]
    print("bounds:", [(p.name, p.bounds) for p in obj.params], flush=True)
    if not a.no_reference:
        print("-- reference points", flush=True)
        for arch, gamma, q in REFERENCE:
            obj.evaluate(arch + [gamma] + ([q, q, q] if a.per_loop else [q]))
    rng = np.random.default_rng(a.seed)
    vals = []
    print(f"-- {a.random} uniformly random points (seed {a.seed})", flush=True)
    for _ in range(a.random):
        x = [float(rng.uniform(lo, hi)) for lo, hi in bounds]
        vals.append(obj.evaluate(x))
    if vals:
        v = np.sort(np.array(vals))
        print(f"random points: best {v[0]:.2f}, quartiles {np.quantile(v, 0.25):.2f} / {np.quantile(v, 0.5):.2f} / {np.quantile(v, 0.75):.2f}, worst {v[-1]:.2f}; "
              f"{int((v < -10.898).sum())} of {len(v)} below the released single knob (-10.90), {int((v < -6.25).sum())} below full depth (-6.25)")
    json.dump({"random_values": vals}, open(os.path.join(a.out, f"probe_random_seed{a.seed}.json"), "w"))


if __name__ == "__main__":
    main()
