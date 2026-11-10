"""
objective/ouro_lcb_router.py
----------------------------
Real-execution router search for the Ouro-2.6B-Thinking looped LM on LiveCodeBench (ICLR 2025), direct mode.

Each evaluation serves the model with TRUE decode-time early exit (skipped recurrence loops are not computed; a token's
cache slots for the skipped loops hold a copy of its exit-step K/V) under the candidate exit rule and returns

    value = -tokens_per_second + penalty_per_pt * max(0, acc_floor - pass@1_pct)       (minimized)

with acc_floor = baseline_pass1_pct - tol_pts: the fastest router that keeps full-depth accuracy, with a linear price on
accuracy lost below the floor.

Two measurements per evaluation, because the machine is shared:
  * accuracy: greedy generation of the `n_items` problems (batch 1, 1,024-token budget, forced ```python block), graded
    against the official tests in a network-less container (lcb_grade.py). Greedy outputs do not depend on the card's
    load, so a generation that ran next to another user's job is still valid, and a finished generation of the same
    thresholds in the work dir is reused.
  * speed: a short calibration run (the first CAL_N problems) whose card is checked for other users' compute before and
    after (nvidia-smi pmon); if another process was computing on the card the calibration is repeated (up to 3 times)
    and the record is flagged. The full-depth speed on the same problems is measured once per card for the speed-up.
    (The 2026-11-09 run measured tokens/s on the whole 342-problem generation and lost 28 % of it to a neighbour's job.)

Search space:
    mode="scalar": one exit threshold q in [0.05, 1.0] (the released gate's single knob)
    mode="steps":  per-step thresholds (q1, q2, q3) in [0.05, 1.0]^3; the token exits after loop s if the gate's
                   cumulative exit probability >= q_s (loop 4 always exits)

Evaluations run as subprocesses of the torchtitan worktree's depth_eval.py; several points evaluate in parallel, one per
GPU in `gpus`; cards can be added while the run is going by appending ids to <work_dir>/gpus.txt. Every evaluation is
recorded in self._eval_records and <work_dir>/evaluations.jsonl (the trajectory for the agent work).
"""
from __future__ import annotations

import getpass
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
    FAIL_VALUE = 50.0
    GEN_TIMEOUT_S = 9 * 3600  # a slow (near full-depth) router over 342 problems at batch 1 takes about 6.5 h
    CAL_N = 30                # problems in the speed calibration (about 15 to 35 min)
    CAL_TIMEOUT_S = 3 * 3600

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
        self.user = getpass.getuser()
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
                f"problems from offset {self.offset}, value = -tok/s (calibrated on {self.CAL_N} problems, contention-checked) "
                f"+ {self.penalty_per_pt} * max(0, {self.acc_floor:.1f} - pass@1 %)")

    @property
    def setup_info(self) -> dict:
        return {"ckpt": self.ckpt, "data": self.data, "mode": self.mode, "n_items": self.n_items, "offset": self.offset,
                "acc_floor_pct": self.acc_floor, "penalty_per_pt": self.penalty_per_pt, "gpus": self.gpus, "cal_n": self.CAL_N}

    # ------------------------------------------------------------------ helpers
    def _thresholds(self, params) -> list[float]:
        vals = [min(1.0, max(0.05, float(v))) for v in params]
        if self.mode == "scalar":
            return [vals[0]] * 3 + [1.0]
        return vals[:3] + [1.0]

    def _label(self, thr) -> str:
        return "q" + "-".join(f"{v:.3f}" for v in thr[:3]) + f"_{int(time.time()) % 100000}"

    @staticmethod
    def _key(thr):
        return [round(float(v), 6) for v in thr]

    def _refresh_gpus(self) -> None:
        """Cards can be added while the run is going: append ids to <work_dir>/gpus.txt (one per line or space-separated)."""
        p = self.work_dir / "gpus.txt"
        if not p.exists():
            return
        try:
            ids = [g for g in p.read_text().replace(",", " ").split() if g.strip()]
        except Exception:
            return
        with self._lock:
            for g in ids:
                if g not in self.gpus:
                    self.gpus.append(g)
                    self._gpu_free.append(g)
                    print(f"[lcb-router] gpu {g} added from gpus.txt", flush=True)

    def _acquire_gpu(self) -> str:
        while True:
            self._refresh_gpus()
            with self._lock:
                if self._gpu_free:
                    return self._gpu_free.pop(0)
            time.sleep(5)

    def _release_gpu(self, g: str) -> None:
        with self._lock:
            self._gpu_free.append(g)

    def _depth_eval(self, thr, gpu, n, out_jsonl, log, timeout):
        """Run depth_eval.py under true early exit; returns the RESULT dict or None."""
        cfg = "decode_early_exit_threshold=" + json.dumps(thr)
        cmd = [self.python_exe, str(self.worktree / "depth_eval.py"), self.ckpt, self.data, "--task", "lcb", "--steps", "4",
               "--n", str(n), "--seed_offset", str(self.offset), "--batch", "1", "--max_new", str(self.max_new),
               "--assistant_prefix", "```python\\n", "--cfg", cfg, "--out", str(out_jsonl)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTORCH_ALLOC_CONF="expandable_segments:True")
        with open(log, "w") as lf:
            subprocess.run(cmd, cwd=str(self.worktree), env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
        result = None
        for line in open(log, errors="replace"):
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
        return result

    def _gpu_contention(self, gpu: str) -> list:
        """Other users' processes computing on this card right now: [(pid, user, sm%)], from nvidia-smi pmon."""
        try:
            out = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-i", gpu], capture_output=True, text=True, timeout=60).stdout
        except Exception:
            return []
        others = []
        for line in out.splitlines():
            parts = line.split()
            if line.startswith("#") or len(parts) < 4 or not parts[1].isdigit() or parts[3] == "-":
                continue
            pid, sm = parts[1], int(parts[3])
            try:
                user = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True, timeout=10).stdout.strip() or "?"
            except Exception:
                user = "?"
            if user != self.user and sm >= 5:
                others.append((pid, user, sm))
        return others

    def _calibrate(self, thr, gpu, run_dir: Path, tag: str):
        """Speed on the first CAL_N problems, repeated while another user computes on the card (up to 3 runs)."""
        best = None
        for attempt in range(3):
            before = self._gpu_contention(gpu)
            res = self._depth_eval(thr, gpu, self.CAL_N, run_dir / f"cal_{tag}_{attempt}.jsonl", run_dir / f"cal_{tag}_{attempt}.log", self.CAL_TIMEOUT_S)
            after = self._gpu_contention(gpu)
            if res is None:
                continue
            clean = not before and not after
            cand = {"tok_s": float(res["tok_s"]), "clean": clean, "attempt": attempt, "others": before + after,
                    "mean_loops_counter": res.get("mean_loops"), "mean_tokens": res.get("mean_tokens")}
            if best is None or cand["tok_s"] > best["tok_s"]:
                best = cand
            if clean:
                break
            print(f"[lcb-router] calibration on gpu {gpu} shared with {before + after}; repeating", flush=True)
            time.sleep(60)
        return best

    def _cal_full(self, gpu: str, run_dir: Path):
        """Full-depth calibration speed on this card (measured once, cached in the work dir)."""
        p = self.work_dir / f"cal_full_gpu{gpu}.json"
        if p.exists():
            try:
                d = json.load(open(p))
                if d.get("clean"):
                    return d
            except Exception:
                pass
        d = self._calibrate([1.0, 1.0, 1.0, 1.0], gpu, run_dir, "full")
        if d:
            json.dump(d, open(p, "w"), indent=1)
        return d

    def _find_graded(self, thr):
        """A finished, graded generation of the same thresholds in the work dir (accuracy does not depend on the card's load)."""
        key = self._key(thr)
        for rec_path in sorted(self.work_dir.glob("q*/record.json")):
            try:
                rec = json.load(open(rec_path))
            except Exception:
                continue
            if self._key(rec.get("thresholds", [])) != key or "pass1_pct" not in rec:
                continue
            if (rec_path.parent / "graded.json").exists():
                return rec
        return None

    def _cached(self, thr):
        """A completed evaluation of the same thresholds in this run's log (e.g. from a crashed run) is reused."""
        p = self.work_dir / "evaluations.jsonl"
        if not p.exists():
            return None
        key = self._key(thr)
        for line in open(p):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") == "ok" and "tok_s_cal" in rec and self._key(rec.get("thresholds", [])) == key:
                return rec
        return None

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, params) -> float:
        """One evaluation. A completed evaluation of the same point is reused; a failed or timed-out generation (e.g. out of memory
        because another job on the shared card grew) is retried once on another card before the worst value is returned, and no
        exception escapes the worker thread (the optimizer would otherwise see None and crash)."""
        thr = self._thresholds(params)
        hit = self._cached(thr)
        if hit is not None:
            rec = dict(hit, status="cached")
            self._eval_records.append(rec)
            print(f"[lcb-router] thr={['%.3f' % v for v in thr[:3]]} cached: pass@1={rec['pass1_pct']:.1f}% tok/s={rec['tok_s']:.2f} value={rec['value']:.3f}", flush=True)
            return float(rec["value"])
        value, ok = self.FAIL_VALUE, False
        gpu = "?"
        for attempt in range(2):
            gpu = self._acquire_gpu()
            try:
                value, ok = self._evaluate_on(params, gpu)
            except Exception as e:  # timeout, grading error, anything: count as a failed generation
                print(f"[lcb-router] evaluation raised on gpu {gpu}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                value, ok = self.FAIL_VALUE, False
            finally:
                self._release_gpu(gpu)
            if ok:
                return value
            if attempt == 0:
                with self._lock:  # prefer a different card for the retry
                    if len(self._gpu_free) > 1 and self._gpu_free[0] == gpu:
                        self._gpu_free.append(self._gpu_free.pop(0))
                print(f"[lcb-router] generation failed on gpu {gpu}; retrying once", flush=True)
        self._eval_records.append({"thresholds": thr, "status": "generation_failed", "value": value, "gpu": gpu})
        return value

    def evaluate_batch(self, points) -> list:
        results = [None] * len(points)

        def worker(k, p):
            try:
                results[k] = self.evaluate(p)
            except Exception as e:  # never hand the optimizer a None
                print(f"[lcb-router] worker {k} raised: {type(e).__name__}: {str(e)[:120]}", flush=True)
                results[k] = self.FAIL_VALUE

        threads = [threading.Thread(target=worker, args=(k, p)) for k, p in enumerate(points)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def _evaluate_on(self, params, gpu: str):
        thr = self._thresholds(params)
        label = self._label(thr)
        run_dir = self.work_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        # 1. accuracy over the n_items problems (reused when the same point was generated and graded before)
        prev = self._find_graded(thr)
        if prev is not None:
            acc_pct, gen = float(prev["pass1_pct"]), {"trunc_rate": prev.get("trunc_rate"), "mean_tokens": prev.get("mean_tokens"),
                                                      "mean_loops": prev.get("mean_loops_counter"), "tok_s": prev.get("tok_s_gen", prev.get("tok_s"))}
            print(f"[lcb-router] thr={['%.3f' % v for v in thr[:3]]} accuracy reused from {prev.get('run_dir')}: {acc_pct:.1f}%", flush=True)
        else:
            out_jsonl = run_dir / "gen.jsonl"
            gen = self._depth_eval(thr, gpu, self.n_items, out_jsonl, run_dir / "gen.log", self.GEN_TIMEOUT_S)
            if gen is None:
                rec = {"thresholds": thr, "status": "generation_failed", "value": self.FAIL_VALUE, "gpu": gpu}
                json.dump(rec, open(run_dir / "record.json", "w"), indent=1)
                return self.FAIL_VALUE, False
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
            if not m:
                rec = {"thresholds": thr, "status": "generation_failed", "value": self.FAIL_VALUE, "gpu": gpu, "note": "grading produced no pass@1"}
                json.dump(rec, open(run_dir / "record.json", "w"), indent=1)
                return self.FAIL_VALUE, False
            acc_pct = 100.0 * float(m.group(1))
            subprocess.run(["cp", str(stage / f"{label}.graded.json"), str(run_dir / "graded.json")], check=False)
        # 2. speed: contention-checked calibration on the first CAL_N problems, plus the card's full-depth reference
        cal = self._calibrate(thr, gpu, run_dir, "thr")
        if cal is None:
            rec = {"thresholds": thr, "status": "generation_failed", "value": self.FAIL_VALUE, "gpu": gpu, "note": "calibration failed"}
            json.dump(rec, open(run_dir / "record.json", "w"), indent=1)
            return self.FAIL_VALUE, False
        full = self._cal_full(gpu, run_dir)
        tok_s = cal["tok_s"]
        value = -tok_s + self.penalty_per_pt * max(0.0, self.acc_floor - acc_pct)
        rec = {"thresholds": thr, "pass1_pct": acc_pct, "tok_s": tok_s, "tok_s_cal": tok_s, "cal_clean": cal["clean"], "cal_attempts": cal["attempt"] + 1,
               "cal_others": cal["others"], "tok_s_full_cal": (full or {}).get("tok_s"), "speedup": (tok_s / full["tok_s"]) if full and full.get("tok_s") else None,
               "tok_s_gen": gen.get("tok_s"), "trunc_rate": gen.get("trunc_rate"), "mean_tokens": gen.get("mean_tokens"),
               "mean_loops_counter": gen.get("mean_loops"), "wall_s": time.time() - t0, "gpu": gpu, "value": value, "status": "ok", "run_dir": str(run_dir)}
        self._eval_records.append(rec)
        json.dump(rec, open(run_dir / "record.json", "w"), indent=1)
        with open(self.work_dir / "evaluations.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[lcb-router] thr={['%.3f' % v for v in thr[:3]]} pass@1={acc_pct:.1f}% tok/s={tok_s:.2f} ({'clean' if cal['clean'] else 'SHARED'}) "
              f"value={value:.3f} ({(time.time()-t0)/60:.0f} min, gpu {gpu})", flush=True)
        return value, True

    def format_for_display(self, value: float) -> float:
        return value
