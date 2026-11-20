"""
objective/ouro_tf_router.py
---------------------------
Teacher-forced proxy of the LiveCodeBench router objective, for searches that need hundreds of evaluations.

The real objective (ouro_lcb_router.py) decodes 342 problems per evaluation, about 3 GPU-hours. Here one evaluation is one
parallel forward over held-out teacher answers (the base model's full-depth answers to LiveCodeBench problems that are NOT
test problems; randrouter/) under the candidate router with the decoder's cache fill (randrouter/core.routed_forward, checked
against token-by-token decoding): about a minute. It returns

    value = -6.25 * 4 / loops + penalty * max(0, floor - agreement %)          (minimised)

loops = executed loops per answer token, agreement = share of answer tokens whose exit readout predicts the full-depth token.
The defaults mirror the real objective's units: floor = 98.45 % is the agreement of the released gate at q = 0.3 on the
validation answers of the base model (the router at the real objective's accuracy floor), and 3.5 per point is the real
penalty (2.0 per pass@1 point) at the observed slope of about 1.7 pass@1 points per agreement point.

Search space: per-loop thresholds (q1, q2, q3) in [0.05, 1.0]^3, as in the real searches.
The system under the router is set by: lora (random-router LoRA state), gate (refitted gate), fill_maps (calibrated fill).
Runs on the card in CUDA_VISIBLE_DEVICES; the model is loaded once per process.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from .base import BaseObjective
from .param import ObjectiveParam

RR = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/randrouter"


class OuroTFRouterObjective(BaseObjective):
    internal_minimize = True

    def __init__(self, *, lora: str | None = None, gate: str | None = None, fill_maps: str | None = None, veto: str | None = None,
                 teacher: str = "/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl", split: str = "val", n_items: int = 99,
                 floor_pct: float = 98.45, penalty_per_pt: float = 3.5, budget_tok: int = 2500, mem_gb: float = 0.0,
                 work_dir: str | None = None, lora_r: int = 32, lora_alpha: float = 64.0, seed: int | None = None, exact: int = 0):
        import torch

        if RR not in sys.path:
            sys.path.insert(0, RR)
        import core
        import tf_eval

        self._core, self._tf, self._torch = core, tf_eval, torch
        if mem_gb:
            torch.cuda.set_per_process_memory_fraction(mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        self.model, self.tok = core.load_model(core.SERVE_DIR)
        if lora:
            core.add_lora(self.model, lora_r, lora_alpha)
            core.load_lora_state(self.model, torch.load(lora, map_location="cpu"))
        if gate:
            g = torch.load(gate, map_location="cpu")
            with torch.no_grad():
                self.model.model.early_exit_gate.weight.copy_(g["weight"].to(self.model.model.early_exit_gate.weight))
                self.model.model.early_exit_gate.bias.copy_(g["bias"].to(self.model.model.early_exit_gate.bias))
        self.fill = core.load_fill_maps(fill_maps) if fill_maps else None
        self.seqs = core.load_teacher(teacher, split=split)[:int(n_items)]
        self.floor, self.penalty, self.budget_tok = float(floor_pct), float(penalty_per_pt), int(budget_tok)
        self.work_dir = Path(work_dir or "/home/jli199/boptim_scratch/randrouter/bo_tf")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.setup = {"lora": lora, "gate": gate, "fill_maps": fill_maps, "split": split, "n_items": len(self.seqs),
                      "floor_pct": self.floor, "penalty_per_pt": self.penalty}
        self.seed = seed
        self.exact = bool(int(exact))
        self.setup["exact_cache"] = self.exact
        self._eval_records: list = []
        self.params = [ObjectiveParam("q1", (0.05, 1.0)), ObjectiveParam("q2", (0.05, 1.0)), ObjectiveParam("q3", (0.05, 1.0))]
        self.veto = veto
        if veto:   # fourth variable: the loop-1 veto threshold on the harm probe's score (3.0 vetoes nothing)
            self.params.append(ObjectiveParam("tau", (-2.5, 3.0)))
            self.setup["veto"] = veto

    @property
    def description(self) -> str:
        return (f"Ouro-2.6B teacher-forced router proxy on {len(self.seqs)} held-out LiveCodeBench answers: "
                f"value = -6.25 * 4 / loops + {self.penalty} * max(0, {self.floor} - agreement %)")

    @property
    def setup_info(self) -> dict:
        return dict(self.setup)

    def measure(self, thr, tau=None):
        """(loops, agreement %) of the router with thresholds thr on the held-out answers, with the decoder's cache."""
        spec = ",".join(f"{float(v):.6f}" for v in thr)
        if self.veto and tau is not None:
            spec += f"|veto={self.veto}@{float(tau):.6f}"
        res = self._tf.evaluate(self.model, self.seqs, [self._tf.parse_router(spec)], self.tok.pad_token_id, self.budget_tok,
                                log=lambda s: None, fill_maps=self.fill, exact=("only" if self.exact else False))[spec]
        return res["loops"], 100.0 * res["agree_exact" if self.exact else "agree_real"], res["share"]

    def evaluate(self, params) -> float:
        thr = [min(1.0, max(0.05, float(v))) for v in params][:3]
        tau = float(params[3]) if self.veto and len(params) > 3 else None
        t0 = time.time()
        loops, agree, share = self.measure(thr, tau)
        value = -6.25 * 4.0 / loops + self.penalty * max(0.0, self.floor - agree)
        rec = {"thresholds": thr, "tau": tau, "loops": loops, "agree_pct": agree, "exit_share": share, "value": value, "wall_s": time.time() - t0,
               "status": "ok"}
        self._eval_records.append(rec)
        with open(self.work_dir / "evaluations.jsonl", "a") as f:
            f.write(json.dumps(dict(rec, **self.setup)) + "\n")
        print(f"[tf-router] thr={['%.3f' % v for v in thr]}{'' if tau is None else ' tau=%.3f' % tau} loops={loops:.3f} agreement={agree:.2f}% value={value:.3f} ({rec['wall_s']:.0f}s)", flush=True)
        return float(value)
