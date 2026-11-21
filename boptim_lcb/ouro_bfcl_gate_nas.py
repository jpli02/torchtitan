"""
objective/ouro_bfcl_gate_nas.py
-------------------------------
Gate NAS on BFCL v3 function calling with real execution: the HumanEval router-NAS experiment on Ouro-2.6B-Thinking (user,
2026-11-21: "only train the gate, like HumanEval"), the BFCL twin of ouro_lcb_gate_nas.py.

Search space (6 dimensions): router depth {0, 1, 2}, width {64, 128, 256}, activation {relu, gelu, silu, tanh}, input LayerNorm
{off, on}, the Stage-II loss threshold adaptive_gamma in [0, 1] and one exit threshold q in [0.1, 1].
One evaluation trains the router from a random initialisation with the Stage-II loss, the model frozen, on cached states of the
released model's own full-depth answers to 600 BFCL LIVE items (randrouter/bfcl_teacher.py; the search evaluates on non-live items),
then serves the model with it (ouro_base_gatelab, decode_router) under true decode-time early exit on the 575 search items with the
official AST checker, in the units of ouro_bfcl_router.py:

    value = 100 * executed loops / 4 + penalty * max(0, floor - accuracy %)          (minimised; full depth = 100)

Every point is decoded (answers are capped at 256 tokens, so a degenerate gate costs at most about an hour); the record keeps the
router file and the fit diagnostics next to the real result.
"""
from __future__ import annotations

import json
import os
import subprocess

from .ouro_bfcl_router import OuroBFCLRouterObjective
from .ouro_lcb_gate_nas import ROUTER_AXES, decode_arch, describe_arch
from .param import ObjectiveParam

RR = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/randrouter"


class OuroBFCLGateNASObjective(OuroBFCLRouterObjective):
    def __init__(self, *, feats: str = "/home/jli199/boptim_scratch/randrouter/nas_feats_bfcl_live.pt",
                 gamma_bounds: tuple[float, float] = (0.0, 1.0), threshold_bounds: tuple[float, float] = (0.1, 1.0),
                 fit_steps: int = 300, fit_seed: int = 0, **kw):
        kw.setdefault("ckpt", "/home/jli199/boptim_scratch/ouro_base_gatelab")
        kw["mode"] = "scalar"
        super().__init__(**kw)
        self.feats, self.fit_steps, self.fit_seed = feats, int(fit_steps), int(fit_seed)
        self.gamma_bounds, self.threshold_bounds = tuple(gamma_bounds), tuple(threshold_bounds)
        self.router_dir = self.work_dir / "routers"
        self.router_dir.mkdir(parents=True, exist_ok=True)
        self.params = [ObjectiveParam(name, (0, len(choices) - 1)) for name, choices in ROUTER_AXES]
        self.params += [ObjectiveParam("adaptive_gamma", self.gamma_bounds), ObjectiveParam("early_exit_threshold", self.threshold_bounds)]

    @property
    def description(self) -> str:
        return (f"Ouro-2.6B gate NAS on BFCL v3 ({self.n_items} items of {self.data_name}): router trained from scratch with the Stage-II loss "
                f"(model frozen) on BFCL live answers, true decode-time early exit; value = 100 * executed loops / 4 + {self.penalty_per_pt} * "
                f"max(0, {self.acc_floor} - accuracy %); axes {[a[0] for a in ROUTER_AXES]} + adaptive_gamma {self.gamma_bounds} + "
                f"threshold {self.threshold_bounds}")

    @property
    def setup_info(self) -> dict:
        info = dict(super().setup_info)
        info.update({"feats": self.feats, "fit_steps": self.fit_steps, "fit_seed": self.fit_seed, "gamma_bounds": self.gamma_bounds,
                     "threshold_bounds": self.threshold_bounds})
        for name, choices in ROUTER_AXES:
            info[name] = choices
        return info

    def _fit(self, spec, gamma, q, path):
        gpu = self.gpus[0]
        cmd = [self.python_exe, os.path.join(RR, "nas_fit_cli.py"), "--spec", json.dumps(spec), "--gamma", repr(float(gamma)), "--q", repr(float(q)),
               "--out", str(path), "--feats", self.feats, "--steps", str(self.fit_steps), "--seed", str(self.fit_seed)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTORCH_ALLOC_CONF="expandable_segments:True")
        out = subprocess.run(cmd, cwd=RR, env=env, capture_output=True, text=True, timeout=3600)
        for line in out.stdout.splitlines():
            if line.startswith("FIT "):
                return json.loads(line[4:])
        print(f"[bfcl-gate-nas] router fit failed: {out.stdout[-300:]} {out.stderr[-500:]}", flush=True)
        return None

    def evaluate(self, params) -> float:
        n = len(ROUTER_AXES)
        spec = decode_arch(params[:n])
        gamma = min(self.gamma_bounds[1], max(self.gamma_bounds[0], float(params[n])))
        q = min(self.threshold_bounds[1], max(self.threshold_bounds[0], float(params[n + 1])))
        arch = describe_arch(spec)
        variant = f"{arch}_g{gamma:.4f}_s{self.fit_seed}"
        path = self.router_dir / f"{variant}.pt"
        self.extra_cfg = {"decode_router": str(path)}      # part of the parent's cache key and record
        if self._cached(self._thresholds([q])) is None:
            if not path.exists():
                rep = self._fit(spec, gamma, q, path)
                if rep is None:
                    self._eval_records.append({"thresholds": [q, q, q, 1.0], "status": "fit_failed", "value": self.FAIL_VALUE, "variant": variant})
                    return self.FAIL_VALUE
                json.dump(rep, open(str(path) + ".fit.json", "w"), indent=1)
                print(f"[bfcl-gate-nas] {arch} gamma={gamma:.4f} q={q:.3f}: trained ({rep['fit']['fit_s']:.0f}s), loop-1 exit share on the "
                      f"training answers {rep['loop1_exit_share_train']:.2f}; decoding", flush=True)
        return super().evaluate([q])
