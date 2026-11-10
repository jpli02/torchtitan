"""
objective/ouro_lcb_router.py
----------------------------
Real-execution router search for the Ouro-2.6B-Thinking looped LM on LiveCodeBench (ICLR 2025), direct mode.

Each evaluation serves the model with TRUE decode-time early exit (skipped recurrence loops are not computed; a token's
cache slots for the skipped loops hold a copy of its exit-step K/V) under the candidate exit rule, generates the first
`n_items` problems of the LiveCodeBench file at batch 1 (greedy, 1,024-token budget, forced ```python code block),
grades every program against the official tests in a network-less container (lcb_grade.py), and returns

    value = -tokens_per_second + penalty_per_pt * max(0, acc_floor - pass@1_pct)       (minimized)

with acc_floor = baseline_pass1_pct - tol_pts: the fastest router that keeps full-depth accuracy, with a linear price
on accuracy lost below the floor. Measured anchors on problems 0..119 (2026-11-03): full depth 22.5 % at 6.23 tok/s;
released gate q=0.7 23.3 % at 7.58; q=0.5 20.0 % at 8.82; q=0.3 21.7 % at 11.12; q=0.2 12.5 % at 14.21 (collapse).

Search space:
    mode="scalar": one exit threshold q in [0.05, 1.0] (the released gate's single knob)
    mode="steps":  per-step thresholds (q1, q2, q3) in [0.05, 1.0]^3; the token exits after loop s if the gate's
                   cumulative exit probability >= q_s (loop 4 always exits)

Evaluations run as subprocesses of the torchtitan worktree's depth_eval.py; several points evaluate in parallel,
one per GPU in `gpus`. Per-evaluation records (pass@1, tok/s, truncation, loops) are kept in self._eval_records
for wandb / trajectory logging.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from .base import BaseObjective
from .param import ObjectiveParam


class OuroLCBRouterObjective(BaseObjective):
    internal_minimize = True  # the value is already a to-be-minimized scalar (like the HumanEval objective)

    def __init__(
        self,
        *,
        worktree: str = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft",
        python_exe: str = "/home/jli199/boptim-agent/.venv/bin/python",
        ckpt: str = "/home/jli199/boptim_scratch/ouro_base_kvshare",
        data: str = "/home/jli199/.claude/jobs/c0d2da0a/tmp/lcb_all.jsonl",
        mode: str = "steps",
        n_items: int = 120,
        offset: int = 0,
        max_new: int = 1024,
        baseline_pass1_pct: float = 22.5,
        tol_pts: float = 1.0,
        penalty_per_pt: float = 2.0,
        gpus: str = "0",
        work_dir: str | None = None,
        stage_dir: str = "/srv/scratch/jli199/lcb_grade_stage_bo",
        seed: int | None = None,
    ):
        self.worktree, self.python_exe, self.ckpt, self.data = Path(worktree), python_exe, ckpt, data
        self.mode, self.n_items, self.offset, self.max_new = mode, int(n_items), int(offset), int(max_new)
        self.acc_floor = float(baseline_pass1_pct) - float(tol_pts)
        self.penalty_per_pt = float(penalty_per_pt)
        self.gpus = [g.strip() for g in str(gpus).split(",") if g.strip()]
        self.work_dir = Path(work_dir or "/home/jli199/boptim_scratch/bo_lcb_router")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir = Path(stage_dir)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.stage_dir, 0o777)
        self.seed = seed
        self._eval_records: list = []
        self._lock = threading.Lock()
        self._gpu_free = list(self.gpus)
        if mode == "scalar":
            self.params = [ObjectiveParam("q", (0.05, 1.0))]
        else:
            self.params = [ObjectiveParam("q1", (0.05, 1.0)), ObjectiveParam("q2", (0.05, 1.0)), ObjectiveParam("q3", (0.05, 1.0))]

    # ------------------------------------------------------------------ metadata
    @property
    def description(self) -> str:
        return (f"Ouro-2.6B LiveCodeBench router search ({self.mode}): true decode-time early exit, n={self.n_items} "
                f"problems from offset {self.offset}, value = -tok/s + {self.penalty_per_pt} * max(0, {self.acc_floor:.1f} - pass@1 %)")

    @property
    def setup_info(self) -> dict:
        return {"ckpt": self.ckpt, "data": self.data, "mode": self.mode, "n_items": self.n_items, "offset": self.offset,
                "acc_floor_pct": self.acc_floor, "penalty_per_pt": self.penalty_per_pt, "gpus": self.gpus}

    # ------------------------------------------------------------------ evaluation
    def _thresholds(self, params) -> list[float]:
        vals = [min(1.0, max(0.05, float(v))) for v in params]
        if self.mode == "scalar":
            return [vals[0]] * 3 + [1.0]
        return vals[:3] + [1.0]

    def _label(self, thr) -> str:
        return "q" + "-".join(f"{v:.3f}" for v in thr[:3]) + f"_{int(time.time()) % 100000}"

    def _acquire_gpu(self) -> str:
        while True:
            with self._lock:
                if self._gpu_free:
                    return self._gpu_free.pop(0)
            time.sleep(5)

    def _release_gpu(self, g: str) -> None:
        with self._lock:
            self._gpu_free.append(g)

    def evaluate(self, params) -> float:
        """One evaluation; a failed generation (e.g. out of memory because another job on the shared card grew) is retried once on
        another card before the worst value is returned, so a transient failure does not poison the surrogate."""
        for attempt in range(2):
            gpu = self._acquire_gpu()
            try:
                value = self._evaluate_on(params, gpu)
            finally:
                self._release_gpu(gpu)
            if self._eval_records and self._eval_records[-1].get("status") == "ok":
                return value
            if attempt == 0:
                self._eval_records.pop()  # drop the failed record; try once more on (preferably) a different card
                with self._lock:
                    if len(self._gpu_free) > 1 and self._gpu_free[0] == gpu:
                        self._gpu_free.append(self._gpu_free.pop(0))
                print(f"[lcb-router] generation failed on gpu {gpu}; retrying once", flush=True)
        return value

    def evaluate_batch(self, points) -> list:
        results = [None] * len(points)

        def worker(k, p):
            results[k] = self.evaluate(p)

        threads = [threading.Thread(target=worker, args=(k, p)) for k, p in enumerate(points)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def _evaluate_on(self, params, gpu: str) -> float:
        thr = self._thresholds(params)
        label = self._label(thr)
        run_dir = self.work_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = run_dir / "gen.jsonl"
        log = run_dir / "gen.log"
        cfg = "decode_early_exit_threshold=" + json.dumps(thr)
        cmd = [self.python_exe, str(self.worktree / "depth_eval.py"), self.ckpt, self.data, "--task", "lcb", "--steps", "4",
               "--n", str(self.n_items), "--seed_offset", str(self.offset), "--batch", "1", "--max_new", str(self.max_new),
               "--assistant_prefix", "```python\\n", "--cfg", cfg, "--out", str(out_jsonl)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTORCH_ALLOC_CONF="expandable_segments:True")
        t0 = time.time()
        with open(log, "w") as lf:
            subprocess.run(cmd, cwd=str(self.worktree), env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=6 * 3600)
        result = None
        for line in open(log, errors="replace"):
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
        if result is None:
            rec = {"thresholds": thr, "status": "generation_failed", "value": 50.0}
            self._eval_records.append(rec)
            json.dump(rec, open(run_dir / "record.json", "w"), indent=1)
            return 50.0
        # grade in the sandbox (staging dir is bind-mounted; the job dir cannot be)
        stage = self.stage_dir
        subprocess.run(["cp", str(self.worktree / "lcb_grade.py"), str(stage / "lcb_grade.py")], check=False)
        if not (stage / Path(self.data).name).exists():
            subprocess.run(["cp", self.data, str(stage / Path(self.data).name)], check=False)
        subprocess.run(["cp", str(out_jsonl), str(stage / f"{label}.jsonl")], check=False)
        grade_log = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--memory", "12g", "--cpus", "4", "-v", f"{stage}:/w", "python:3.12-slim",
             "python", "/w/lcb_grade.py", f"/w/{Path(self.data).name}", f"/w/{label}.jsonl", f"/w/{label}.graded.json"],
            capture_output=True, text=True, timeout=3600,
        ).stdout
        m = re.search(r"pass@1=([0-9.]+)", grade_log)
        acc_pct = 100.0 * float(m.group(1)) if m else 0.0
        subprocess.run(["cp", str(stage / f"{label}.graded.json"), str(run_dir / "graded.json")], check=False)
        tok_s = float(result["tok_s"])
        value = -tok_s + self.penalty_per_pt * max(0.0, self.acc_floor - acc_pct)
        rec = {"thresholds": thr, "pass1_pct": acc_pct, "tok_s": tok_s, "trunc_rate": result.get("trunc_rate"),
               "mean_tokens": result.get("mean_tokens"), "mean_loops_counter": result.get("mean_loops"), "wall_s": time.time() - t0,
               "gpu": gpu, "value": value, "status": "ok", "run_dir": str(run_dir)}
        self._eval_records.append(rec)
        json.dump(rec, open(run_dir / "record.json", "w"), indent=1)
        with open(self.work_dir / "evaluations.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[lcb-router] thr={['%.3f' % v for v in thr[:3]]} pass@1={acc_pct:.1f}% tok/s={tok_s:.2f} value={value:.3f} ({(time.time()-t0)/60:.0f} min, gpu {gpu})", flush=True)
        return value

    def format_for_display(self, value: float) -> float:
        return value
