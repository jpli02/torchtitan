"""
objective/ouro_lcb_gate_nas.py
------------------------------
Gate NAS on LiveCodeBench with real execution: the HumanEval router-NAS experiment (objective/ouro_router_nas.py, Ouro-1.4B) on
Ouro-2.6B-Thinking, as asked by the user on 2026-11-21 ("only train the gate, like HumanEval").

Search space (6 dimensions, the same as ouro_router_nas): router depth {0, 1, 2}, width {64, 128, 256}, activation {relu, gelu,
silu, tanh}, input LayerNorm {off, on} (each a continuous dimension rounded to the nearest choice; width and activation do nothing
at depth 0), the Stage-II loss threshold adaptive_gamma in [0, 1] and the exit threshold q in [0.1, 1] (one value for all loops,
like the released gate's knob).

One evaluation:
  1. trains the router from a RANDOM initialisation with the Stage-II loss while the language model stays frozen
     (randrouter/nas_fit_cli.py on the evaluation's card: cached normed states of the base model's own full-depth answers to 172
     LiveCodeBench v1-v4 training problems, 66,474 positions; label w = sigmoid(50 (CE improvement - gamma)); 300 AdamW steps,
     batch 8,192, seed 0, so the same point always gives the same router);
  2. serves the released model with that router as its exit gate (serving dir ouro_base_gatelab, option decode_router) under
     true decode-time early exit and scores the 342 test problems with the official tests (the machinery and units of
     ouro_lcb_router.py with cost="loops"):

     value = -6.25 * 4 / executed loops per token + penalty * max(0, floor - pass@1 %)          (minimised; full depth = -6.25)

Two degenerate ends are scored without decoding, from runs that were measured for real (the objective stays deterministic):
  * a router that sends at least 95 % of the training positions out after loop 1 = fixed depth 1 (5.6 % pass@1 at 1.0 loops);
  * a router whose teacher-forced loops on the 34 held-out answers are at least 3.98 = full depth (20.5 % at 4.0 loops).
Every record keeps the router file, the fit diagnostics and the teacher-forced proxy numbers (loops, the two kinds of token error,
calibrated pass@1) next to the real result.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .ouro_lcb_router import OuroLCBRouterObjective
from .param import ObjectiveParam

RR = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/randrouter"
ROUTER_AXES: list[tuple[str, list]] = [
    ("router_layers", [0, 1, 2]),
    ("router_hidden", [64, 128, 256]),
    ("router_act", ["relu", "gelu", "silu", "tanh"]),
    ("router_norm", [False, True]),
]


def decode_arch(params) -> dict:
    spec = {}
    for (name, choices), v in zip(ROUTER_AXES, params):
        idx = max(0, min(len(choices) - 1, int(round(float(v)))))
        spec[name.removeprefix("router_")] = choices[idx]
    if spec["layers"] == 0:          # width and activation do not exist without hidden layers
        spec = {"layers": 0, "norm": spec["norm"]}
    return spec


def describe_arch(spec: dict) -> str:
    if int(spec.get("layers", 0)) == 0:
        return "linear-ln" if spec.get("norm") else "linear"
    return f"L{spec['layers']}-H{spec['hidden']}-{spec['act']}-{'ln' if spec.get('norm') else 'noln'}"


class OuroLCBGateNASObjective(OuroLCBRouterObjective):
    DEPTH1 = (5.6, 1.0)        # fixed depth 1 on the 342 problems (bench_lcb342_k1): pass@1 %, loops
    FULL = (20.5, 4.0)         # full depth on the 342 problems

    def __init__(self, *, feats: str = "/home/jli199/boptim_scratch/randrouter/nas_feats_train.pt",
                 gamma_bounds: tuple[float, float] = (0.0, 1.0), threshold_bounds: tuple[float, float] = (0.1, 1.0),
                 fit_steps: int = 300, fit_seed: int = 0, shortcut: int = 1, proxy: int = 1, **kw):
        kw.setdefault("ckpt", "/home/jli199/boptim_scratch/ouro_base_gatelab")
        kw["cost"] = "loops"
        kw["mode"] = "scalar"
        super().__init__(**kw)
        self.feats, self.fit_steps, self.fit_seed = feats, int(fit_steps), int(fit_seed)
        self.gamma_bounds, self.threshold_bounds = tuple(gamma_bounds), tuple(threshold_bounds)
        self.shortcut, self.proxy = bool(int(shortcut)), bool(int(proxy))
        self.router_dir = self.work_dir / "routers"
        self.router_dir.mkdir(parents=True, exist_ok=True)
        self.params = [ObjectiveParam(name, (0, len(choices) - 1)) for name, choices in ROUTER_AXES]
        self.params += [ObjectiveParam("adaptive_gamma", self.gamma_bounds), ObjectiveParam("early_exit_threshold", self.threshold_bounds)]
        self._cur = None

    @property
    def description(self) -> str:
        return (f"Ouro-2.6B gate NAS on LiveCodeBench ({self.n_items} problems of {Path(self.data).name}): router trained from scratch with the "
                f"Stage-II loss (model frozen), true decode-time early exit; value = -6.25 * 4 / executed loops + {self.penalty_per_pt} * "
                f"max(0, {self.acc_floor:.1f} - pass@1 %); axes {[a[0] for a in ROUTER_AXES]} + adaptive_gamma {self.gamma_bounds} + "
                f"threshold {self.threshold_bounds}")

    @property
    def setup_info(self) -> dict:
        info = dict(super().setup_info)
        info.update({"feats": self.feats, "fit_steps": self.fit_steps, "fit_seed": self.fit_seed, "gamma_bounds": self.gamma_bounds,
                     "threshold_bounds": self.threshold_bounds, "shortcut": self.shortcut, "proxy": self.proxy})
        for name, choices in ROUTER_AXES:
            info[name] = choices
        return info

    # hooks of the parent
    def _variant(self):
        return None if self._cur is None else self._cur["variant"]

    def _extra_cfg(self) -> list[str]:
        return [] if self._cur is None else [f"decode_router={self._cur['router']}"]

    def _rec_extra(self) -> dict:
        if self._cur is None:
            return {}
        return {k: self._cur.get(k) for k in ("router_spec", "router_arch", "gamma", "router", "fit_report", "params")}

    def _fit(self, spec, gamma, q, path):
        gpu = self.gpus[0]
        cmd = [self.python_exe, os.path.join(RR, "nas_fit_cli.py"), "--spec", json.dumps(spec), "--gamma", repr(float(gamma)), "--q", repr(float(q)),
               "--out", str(path), "--feats", self.feats, "--steps", str(self.fit_steps), "--seed", str(self.fit_seed)]
        if self.proxy:
            cmd.append("--proxy")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTORCH_ALLOC_CONF="expandable_segments:True")
        out = subprocess.run(cmd, cwd=RR, env=env, capture_output=True, text=True, timeout=3600)
        for line in out.stdout.splitlines():
            if line.startswith("FIT "):
                return json.loads(line[4:])
        print(f"[gate-nas] router fit failed: {out.stdout[-300:]} {out.stderr[-500:]}", flush=True)
        return None

    def _write(self, rec):
        self._eval_records.append(rec)
        with open(self.work_dir / "evaluations.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def evaluate(self, params) -> float:
        n = len(ROUTER_AXES)
        spec = decode_arch(params[:n])
        gamma = min(self.gamma_bounds[1], max(self.gamma_bounds[0], float(params[n])))
        q = min(self.threshold_bounds[1], max(self.threshold_bounds[0], float(params[n + 1])))
        thr = [q, q, q, 1.0]
        arch = describe_arch(spec)
        variant = f"{arch}_g{gamma:.4f}_s{self.fit_seed}"
        path = self.router_dir / f"{variant}.pt"
        self._cur = {"variant": variant, "router": str(path), "router_spec": spec, "router_arch": arch, "gamma": gamma,
                     "params": [float(v) for v in params]}
        hit = self._cached(thr)
        if hit is not None:
            return super().evaluate([q])          # the parent returns the cached value
        rep = self._fit(spec, gamma, q, path)
        if rep is None:
            rec = {"thresholds": thr, "status": "fit_failed", "value": self.FAIL_VALUE, "variant": variant, **self._rec_extra()}
            self._write(rec)
            return self.FAIL_VALUE
        self._cur["fit_report"] = rep
        tag = None
        if self.shortcut and rep.get("loop1_exit_share_train", 0.0) >= 0.95:
            (p1, loops), tag = self.DEPTH1, "fixed depth 1"
        elif self.shortcut and rep.get("proxy", {}).get("loops", 0.0) >= 3.98:
            (p1, loops), tag = self.FULL, "full depth"
        if tag:
            value = -6.25 * 4.0 / loops + self.penalty_per_pt * max(0.0, self.acc_floor - p1)
            rec = {"thresholds": thr, "status": "shortcut", "shortcut": tag, "pass1_pct": p1, "loops_eff": loops, "value": value,
                   "fill_maps": self.fill_maps, "cost": "loops", "variant": variant, **self._rec_extra()}
            self._write(rec)
            print(f"[gate-nas] {arch} gamma={gamma:.4f} q={q:.3f}: scored as {tag} without decoding (value {value:.3f})", flush=True)
            return float(value)
        pr = rep.get("proxy", {})
        print(f"[gate-nas] {arch} gamma={gamma:.4f} q={q:.3f}: trained ({rep['fit']['fit_s']:.0f}s), loop-1 exit share {rep['loop1_exit_share_train']:.2f}, "
              f"proxy loops {pr.get('loops', float('nan')):.2f}, pass@1~{pr.get('pass1_hat', float('nan')):.1f}; decoding", flush=True)
        return super().evaluate([q])
