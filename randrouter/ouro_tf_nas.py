"""
objective/ouro_tf_nas.py
------------------------
Fast replica of the router NAS objective (objective/ouro_router_nas.py) for optimizer comparisons that need many seeds.

Same search space as the real objective (6 dimensions): the Tier-1 router architecture (depth, width, activation, input
LayerNorm; each axis a continuous dimension rounded to the nearest choice, as in ouro_router_nas.decode_arch), the Stage-II
loss threshold adaptive_gamma in [0, 1] and the inference exit threshold in [0.1, 1].
Same procedure: every candidate router is trained from a RANDOM INITIALISATION with the Stage-II loss while the language
model stays frozen, then the model is run with early exit under that router.
What makes it fast (about 20 s per evaluation instead of a training run plus a benchmark run):
  - the frozen model's states are cached once (randrouter/nas_feats.py), so Stage II is a few seconds (randrouter/nas_fit.py);
  - the score is a teacher-forced proxy instead of generated code: the base model's own full-depth answers to held-out
    LiveCodeBench problems are replayed under the router with the decoder's cache behaviour, in the real objective's units,

    value = -6.25 * 4 / loops + 2.0 * max(0, 18.3 - predicted pass@1 %)          (minimised)
    predicted pass@1 = c - alpha * own - beta * cache                           (randrouter/proxy_value.py, calib_pass1.py)

    loops = executed loops per answer token; own = % of answer tokens whose exit readout is wrong even with an exact cache
    (the token left too early); cache = % right with an exact cache and wrong with the decoder's cache. The coefficients are
    fitted on the routers that were run for real on the 342 test problems. calib=0 falls back to the older penalty on raw
    token agreement (floor - agreement %, 3.5 per point), whose optimum is the degenerate "every token leaves after loop 1".
Deterministic: the router's initialisation and minibatch order are seeded with the objective seed.
Options: per_loop=1 replaces the single exit threshold by three per-loop thresholds (8 dimensions); exact=1 scores with an
exact cache (what the real NAS objective's no-KV-cache evaluation path does).
Runs on the card in CUDA_VISIBLE_DEVICES; the model and the features are loaded once per process.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .base import BaseObjective
from .param import ObjectiveParam

RR = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/randrouter"

ROUTER_AXES: list[tuple[str, list]] = [
    ("router_layers", [0, 1, 2]),
    ("router_hidden", [64, 128, 256]),
    ("router_act", ["relu", "gelu", "silu", "tanh"]),
    ("router_norm", [False, True]),
]


def decode_arch(params, axes=ROUTER_AXES) -> dict:
    spec = {}
    for (name, choices), v in zip(axes, params):
        idx = max(0, min(len(choices) - 1, int(round(float(v)))))
        spec[name.removeprefix("router_")] = choices[idx]
    return spec


class OuroTFNASObjective(BaseObjective):
    internal_minimize = True

    def __init__(self, *, feats: str = "/home/jli199/boptim_scratch/randrouter/nas_feats_train.pt",
                 teacher: str = "/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl", split: str = "val", n_items: int = 99,
                 floor_pct: float = 98.45, penalty_per_pt: float = 3.5, budget_tok: int = 2500, mem_gb: float = 0.0,
                 work_dir: str | None = None, seed: int | None = None, exact: int = 0, per_loop: int = 0, steps: int = 300,
                 lr: float = 1e-3, k: float = 50.0, gamma_bounds: tuple[float, float] = (0.0, 1.0),
                 threshold_bounds: tuple[float, float] = (0.1, 1.0), calib: int = 1):
        import torch

        if RR not in sys.path:
            sys.path.insert(0, RR)
        import core
        import nas_fit
        import proxy_value
        import tf_eval

        self._core, self._tf, self._fit, self._torch, self._pv = core, tf_eval, nas_fit, torch, proxy_value
        self.calib = proxy_value.load_calib() if int(calib) else None
        if mem_gb:
            torch.cuda.set_per_process_memory_fraction(mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        self.model, self.tok = core.load_model(core.SERVE_DIR)
        self.feats = nas_fit.load_feats(feats)
        self.seqs = core.load_teacher(teacher, split=split)[:int(n_items)]
        self.floor, self.penalty, self.budget_tok = float(floor_pct), float(penalty_per_pt), int(budget_tok)
        self.work_dir = Path(work_dir or "/home/jli199/boptim_scratch/randrouter/bo_tfnas")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.seed = 0 if seed is None else int(seed)
        self.exact, self.per_loop = bool(int(exact)), bool(int(per_loop))
        self.steps, self.lr, self.k = int(steps), float(lr), float(k)
        self.gamma_bounds, self.threshold_bounds = tuple(gamma_bounds), tuple(threshold_bounds)
        self.setup = {"feats": feats, "split": split, "n_items": len(self.seqs), "floor_pct": self.floor, "penalty_per_pt": self.penalty,
                      "exact_cache": self.exact, "per_loop": self.per_loop, "steps": self.steps, "lr": self.lr, "k": self.k,
                      "fit_seed": self.seed, "n_train_positions": int(self.feats["h"].shape[0]), "calib": self.calib}
        self._eval_records: list = []
        self._fit_cache: dict = {}
        self.params = [ObjectiveParam(name, (0, len(choices) - 1)) for name, choices in ROUTER_AXES]
        self.params.append(ObjectiveParam("adaptive_gamma", self.gamma_bounds))
        if self.per_loop:
            self.params += [ObjectiveParam(f"q{i}", (0.05, 1.0)) for i in (1, 2, 3)]
        else:
            self.params.append(ObjectiveParam("early_exit_threshold", self.threshold_bounds))

    @property
    def description(self) -> str:
        head = (f"Ouro-2.6B router NAS, fast replica: router trained from a random initialisation on cached states, scored by the "
                f"teacher-forced proxy on {len(self.seqs)} held-out LiveCodeBench answers; ")
        if self.calib:
            return head + (f"value = -6.25 * 4 / loops + {self._pv.PENALTY_PER_PASS1_PT} * max(0, {self._pv.FLOOR_PASS1} - predicted pass@1 %), "
                           f"predicted pass@1 = {self.calib['c']:.2f} - {self.calib['alpha']:.2f} * own - {self.calib['beta']:.2f} * cache")
        return head + f"value = -6.25 * 4 / loops + {self.penalty} * max(0, {self.floor} - agreement %)"

    @property
    def setup_info(self) -> dict:
        info = dict(self.setup)
        for name, choices in ROUTER_AXES:
            info[name] = choices
        return info

    def _router(self, spec: dict, gamma: float):
        key = (spec["layers"], spec["hidden"], spec["act"], bool(spec["norm"]), round(float(gamma), 6))
        if spec["layers"] == 0:                       # width and activation do not exist in a router without hidden layers
            key = (0, None, None, bool(spec["norm"]), round(float(gamma), 6))
        if key not in self._fit_cache:
            if len(self._fit_cache) > 8:
                self._fit_cache.clear()
            router, diag = self._fit.fit_router(self.feats, spec, gamma, k=self.k, steps=self.steps, lr=self.lr, seed=self.seed)
            self._fit_cache[key] = (router.to(self._torch.bfloat16), diag)
        return self._fit_cache[key]

    def evaluate(self, params) -> float:
        t0 = time.time()
        n_axes = len(ROUTER_AXES)
        spec = decode_arch(params[:n_axes])
        gamma = min(self.gamma_bounds[1], max(self.gamma_bounds[0], float(params[n_axes])))
        if self.per_loop:
            thr = [min(1.0, max(0.05, float(v))) for v in params[n_axes + 1:n_axes + 4]]
        else:
            q = min(self.threshold_bounds[1], max(self.threshold_bounds[0], float(params[n_axes + 1])))
            thr = [q, q, q]
        router, diag = self._router(spec, gamma)
        rspec = ",".join(f"{v:.6f}" for v in thr)
        # with the calibrated value the same exits are replayed with an exact cache as well (the two kinds of error)
        res = self._tf.evaluate(self.model, self.seqs, [self._tf.parse_router(rspec)], self.tok.pad_token_id, self.budget_tok, gate=router,
                                log=lambda s: None, exact=("only" if self.exact else bool(self.calib)))[rspec]
        loops, agree = res["loops"], 100.0 * res["agree_exact" if self.exact else "agree_real"]
        own = 100.0 - 100.0 * res["agree_exact"]
        cache = 0.0 if self.exact else 100.0 * (res["agree_exact"] - res["agree_real"])
        if self.calib:
            value, p1 = self._pv.value(loops, own, cache, self.calib)
        else:
            value, p1 = -6.25 * 4.0 / loops + self.penalty * max(0.0, self.floor - agree), None
        rec = {"router_spec": spec, "router_arch": self._fit.describe_arch(spec), "gamma": gamma, "thresholds": thr, "loops": loops,
               "agree_pct": agree, "own_err_pct": own, "cache_err_pct": cache, "pass1_hat": p1, "exit_share": res["share"], "value": value,
               "fit": diag, "wall_s": time.time() - t0, "status": "ok", "params": [float(v) for v in params]}
        self._eval_records.append(rec)
        with open(self.work_dir / "evaluations.jsonl", "a") as f:
            f.write(json.dumps(dict(rec, **self.setup)) + "\n")
        print(f"[tf-nas] {rec['router_arch']} gamma={gamma:.4f} thr={['%.3f' % v for v in thr]} loops={loops:.3f} agreement={agree:.2f}% "
              f"(own {own:.2f}, cache {cache:.2f})" + ("" if p1 is None else f" pass@1~{p1:.1f}") +
              f" value={value:.3f} ({rec['wall_s']:.0f}s, fit {diag['fit_s']:.1f}s)", flush=True)
        return float(value)
