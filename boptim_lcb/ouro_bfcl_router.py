"""
objective/ouro_bfcl_router.py
-----------------------------
Router search for the Ouro-2.6B-Thinking looped LM on BFCL v3 single-turn function calling (ICML 2025), under REAL decode-time
early exit (skipped recurrence loops are not computed).

One evaluation of a candidate exit rule:
  * generates the search set (official BFCL prompt, the call list forced to start with "[", greedy, batch 1, 256 tokens) as a few
    concurrent batch-1 processes on one card (batch-1 greedy outputs are exactly reproducible, so shards and card load do not matter);
  * grades the calls with the official AST checker inside the sandbox image bfcl-grader (bfcl_tool.py grade);
  * reads the compute from the model's own count of loops executed per decoded token (deterministic, unlike wall-clock tokens/s on
    this shared machine, which varied by 10 to 30 % between identical runs).

    value = 100 * executed_loops / 4  +  penalty_per_pt * max(0, acc_floor - accuracy_pct)          (minimised)

so full depth scores 100 and a router that halves the decode compute at no accuracy cost scores 50. accuracy = share of items whose
call passes the checker (item-weighted); acc_floor = full-depth accuracy on the search set minus tol_pts (one standard error).

Search space: mode="steps": per-loop exit thresholds (q1, q2, q3) in [0.05, 1]^3 (a token exits after loop s when the gate's
cumulative exit probability reaches q_s; loop 4 always exits); mode="scalar": one threshold (the released gate's knob).

Finished evaluations are reused from <work_dir>/evaluations.jsonl; cards can be added while running through <work_dir>/gpus.txt;
a card without MIN_FREE_MIB free memory is skipped; a failed evaluation is retried once and never raises into the optimizer.
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


class OuroBFCLRouterObjective(BaseObjective):
    internal_minimize = True
    FAIL_VALUE = 200.0
    GEN_TIMEOUT_S = 5 * 3600
    MIN_FREE_MIB = 12000
    MAX_ATTEMPTS = 4   # tries per evaluation before the failure value is reported to the optimizer
    SHARD_MIB = 9500   # one batch-1 process with a 2k-token prompt: model 5.5 GB + cache
    PREFILL_TOKEN_EQUIV = 8.0  # prompt prefill of one item, in units of one decoded token (shard balancing only)

    def __init__(
        self,
        *,
        worktree: str = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft",
        python_exe: str = "/home/jli199/boptim-agent/.venv/bin/python",
        ckpt: str = "/home/jli199/boptim_scratch/ouro_base_kvshare",
        data_name: str = "bfcl_search.jsonl",
        stage_dir: str = "/srv/scratch/jli199/bfcl_stage",
        mode: str = "steps",
        n_items: int = 575,
        max_new: int = 256,
        baseline_acc_pct: float | None = None,
        tol_pts: float = 1.7,
        penalty_per_pt: float = 5.0,
        gpus: str = "3",
        shards: int = 2,
        work_dir: str | None = None,
        seed: int | None = None,
    ):
        self.worktree, self.python_exe, self.ckpt = Path(worktree), python_exe, ckpt
        self.stage_dir, self.data_name = Path(stage_dir), data_name
        self.data = str(self.stage_dir / data_name)
        self.mode, self.n_items, self.max_new = mode, int(n_items), int(max_new)
        self.tol_pts, self.penalty_per_pt = float(tol_pts), float(penalty_per_pt)
        self.acc_floor = None if baseline_acc_pct is None else float(baseline_acc_pct) - self.tol_pts
        self.gpus = [g.strip() for g in str(gpus).split(",") if g.strip()]
        self.shards = int(shards)
        self.work_dir = Path(work_dir or "/home/jli199/boptim_scratch/bo_bfcl_router")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self._eval_records: list = []
        self._lock = threading.Lock()
        self._gpu_free = list(self.gpus)
        names = ["q"] if mode == "scalar" else ["q1", "q2", "q3"]
        self.params = [ObjectiveParam(n, (0.05, 1.0)) for n in names]

    # ------------------------------------------------------------------ metadata
    @property
    def description(self) -> str:
        return (f"Ouro-2.6B BFCL v3 router search ({self.mode}): real decode-time early exit on {self.n_items} items of {self.data_name}, "
                f"value = 100 * executed loops / 4 + {self.penalty_per_pt} * max(0, {self.acc_floor} - accuracy %)")

    @property
    def setup_info(self) -> dict:
        return {"ckpt": self.ckpt, "data": self.data, "mode": self.mode, "n_items": self.n_items, "acc_floor_pct": self.acc_floor,
                "penalty_per_pt": self.penalty_per_pt, "gpus": self.gpus, "shards": self.shards}

    # ------------------------------------------------------------------ helpers
    def value_of(self, acc_pct: float, exec_loops: float) -> float:
        floor = self.acc_floor if self.acc_floor is not None else -1e9
        return 100.0 * exec_loops / 4.0 + self.penalty_per_pt * max(0.0, floor - acc_pct)

    def _thresholds(self, params) -> list[float]:
        vals = [min(1.0, max(0.05, float(v))) for v in params]
        return ([vals[0]] * 3 if self.mode == "scalar" else vals[:3]) + [1.0]

    @staticmethod
    def _key(thr):
        return [round(float(v), 6) for v in thr]

    def _label(self, thr) -> str:
        return "q" + "-".join(f"{v:.3f}" for v in thr[:3]) + f"_{int(time.time() * 10) % 1000000}"

    @staticmethod
    def _free_mib(gpu: str) -> int:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            return int(out.split()[0])
        except Exception:
            return 0

    def _refresh_gpus(self) -> None:
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
                    print(f"[bfcl-router] gpu {g} added from gpus.txt", flush=True)

    def _acquire_gpu(self) -> str:
        warned = set()
        while True:
            self._refresh_gpus()
            with self._lock:
                for _ in range(len(self._gpu_free)):
                    g = self._gpu_free.pop(0)
                    if self._free_mib(g) >= self.MIN_FREE_MIB:
                        return g
                    self._gpu_free.append(g)
                    if g not in warned:
                        warned.add(g)
                        print(f"[bfcl-router] gpu {g} has under {self.MIN_FREE_MIB} MiB free; waiting for room", flush=True)
            time.sleep(30)

    def _release_gpu(self, g: str) -> None:
        with self._lock:
            self._gpu_free.append(g)

    def _cached(self, thr):
        p = self.work_dir / "evaluations.jsonl"
        if not p.exists():
            return None
        key = self._key(thr)
        for line in open(p):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") == "ok" and self._key(rec.get("thresholds", [])) == key:
                return rec
        return None

    def _bounds(self, k: int):
        """Contiguous shard ranges with about equal work. The items are ordered by category and the later categories need several
        calls per answer, so equal counts leave the last shard with twice the work of the first. When <work_dir>/item_cost.json
        exists (generated tokens per item of a finished evaluation, written below) the cut points equalise cumulative cost instead.
        Batch-1 greedy outputs do not depend on where the cuts fall."""
        n = self.n_items
        equal = [((n * j) // k, (n * (j + 1)) // k) for j in range(k)]
        try:
            cost = json.loads((self.work_dir / "item_cost.json").read_text())
        except Exception:
            return equal
        if k < 2 or len(cost) != n:
            return equal
        total, run, cuts = float(sum(cost)), 0.0, [0]
        for i, c in enumerate(cost):
            run += c
            if len(cuts) < k and run >= total * len(cuts) / k:
                cuts.append(i + 1)
        cuts = cuts[:k] + [n]
        bounds = [(cuts[j], cuts[j + 1]) for j in range(len(cuts) - 1)]
        return bounds if len(bounds) == k and all(hi > lo for lo, hi in bounds) else equal

    def _save_item_cost(self, run_dir) -> None:
        """Per-item cost for _bounds from a finished evaluation: generated tokens plus a constant for the prompt prefill."""
        p = self.work_dir / "item_cost.json"
        if p.exists():
            return
        try:
            cost = [float(json.loads(line)["ntok"]) + self.PREFILL_TOKEN_EQUIV for line in open(run_dir / "gen.jsonl") if line.strip() and '"summary"' not in line[:12]]
            if len(cost) == self.n_items:
                p.write_text(json.dumps(cost))
        except Exception:
            pass

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, params) -> float:
        thr = self._thresholds(params)
        hit = self._cached(thr)
        if hit is not None:
            value = self.value_of(hit["acc_pct"], hit["exec_loops"])
            self._eval_records.append(dict(hit, status="cached", value=value))
            print(f"[bfcl-router] thr={['%.3f' % v for v in thr[:3]]} cached: acc={hit['acc_pct']:.1f}% loops={hit['exec_loops']:.3f} value={value:.2f}", flush=True)
            return value
        value, ok, gpu = self.FAIL_VALUE, False, "?"
        for attempt in range(self.MAX_ATTEMPTS):
            gpu = self._acquire_gpu()
            try:
                value, ok = self._evaluate_on(thr, gpu)
            except Exception as e:
                print(f"[bfcl-router] evaluation raised on gpu {gpu}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                value, ok = self.FAIL_VALUE, False
            finally:
                self._release_gpu(gpu)
            if ok:
                return value
            if attempt < self.MAX_ATTEMPTS - 1:
                # a failure here is the shared machine (a card taken mid-run, a killed process), not the router: wait and try again
                with self._lock:
                    if len(self._gpu_free) > 1 and self._gpu_free[0] == gpu:
                        self._gpu_free.append(self._gpu_free.pop(0))
                print(f"[bfcl-router] evaluation failed on gpu {gpu}; retry {attempt + 1} of {self.MAX_ATTEMPTS - 1}", flush=True)
                time.sleep(120 * (attempt + 1))
        self._eval_records.append({"thresholds": thr, "status": "generation_failed", "value": value, "gpu": gpu})
        return value

    def evaluate_batch(self, points) -> list:
        results = [None] * len(points)

        def worker(k, p):
            try:
                results[k] = self.evaluate(p)
            except Exception as e:
                print(f"[bfcl-router] worker {k} raised: {type(e).__name__}: {str(e)[:120]}", flush=True)
                results[k] = self.FAIL_VALUE

        threads = [threading.Thread(target=worker, args=(k, p)) for k, p in enumerate(points)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def _evaluate_on(self, thr, gpu: str):
        label = self._label(thr)
        run_dir = self.work_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        k = max(1, min(self.shards, self._free_mib(gpu) // self.SHARD_MIB))
        bounds = self._bounds(k)
        cfg = "decode_early_exit_threshold=" + json.dumps(thr)
        procs = []
        for j, (lo, hi) in enumerate(bounds):
            cmd = [self.python_exe, str(self.worktree / "depth_eval.py"), self.ckpt, self.data, "--task", "bfcl", "--steps", "4",
                   "--n", str(hi - lo), "--seed_offset", str(lo), "--batch", "1", "--max_new", str(self.max_new),
                   "--assistant_prefix", "[", "--cfg", cfg, "--out", str(run_dir / f"gen_{j}.jsonl")]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTORCH_ALLOC_CONF="expandable_segments:True")
            procs.append(subprocess.Popen(cmd, cwd=str(self.worktree), env=env, stdout=open(run_dir / f"gen_{j}.log", "w"), stderr=subprocess.STDOUT))
        for p in procs:
            try:
                p.wait(timeout=max(60, self.GEN_TIMEOUT_S - (time.time() - t0)))
            except subprocess.TimeoutExpired:
                for q in procs:
                    q.kill()
                raise
        results = []
        for j in range(k):
            res = None
            for line in open(run_dir / f"gen_{j}.log", errors="replace"):
                if line.startswith("RESULT "):
                    res = json.loads(line[len("RESULT "):])
            if res is None:
                json.dump({"thresholds": thr, "status": "generation_failed", "gpu": gpu, "shard": j}, open(run_dir / "record.json", "w"))
                return self.FAIL_VALUE, False
            results.append(res)
        with open(run_dir / "gen.jsonl", "w") as out:
            out.write(json.dumps({"summary": {"mean_loops": 0.0}}) + "\n")
            for j in range(k):
                for line in open(run_dir / f"gen_{j}.jsonl"):
                    if line.strip() and '"summary"' not in line[:12]:
                        out.write(line if line.endswith("\n") else line + "\n")
        # executed loops per decoded token (a shard without the counter ran at full depth)
        n_tok = sum(r.get("exec_n") or r["mean_tokens"] * r["n"] for r in results)
        exec_loops = sum((r.get("exec_loops") or 4.0) * (r.get("exec_n") or r["mean_tokens"] * r["n"]) for r in results) / max(1.0, n_tok)
        # official AST checker in the sandbox
        subprocess.run(["cp", str(self.worktree / "bfcl_tool.py"), str(self.stage_dir / "bfcl_tool.py")], check=False)
        subprocess.run(["cp", str(run_dir / "gen.jsonl"), str(self.stage_dir / f"bo_{label}.jsonl")], check=False)
        grade_log = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--memory", "8g", "--cpus", "4", "-v", f"{self.stage_dir}:/w", "bfcl-grader",
             "python", "/w/bfcl_tool.py", "grade", f"/w/{self.data_name}", f"/w/bo_{label}.jsonl", f"/w/bo_{label}.graded.json"],
            capture_output=True, text=True, timeout=1800).stdout
        m = re.search(r"GRADED n=(\d+) overall=([0-9.]+) micro=([0-9.]+)", grade_log)
        if not m or int(m.group(1)) != self.n_items:
            json.dump({"thresholds": thr, "status": "generation_failed", "gpu": gpu, "note": "grading: " + grade_log[-200:]}, open(run_dir / "record.json", "w"))
            return self.FAIL_VALUE, False
        subprocess.run(["cp", str(self.stage_dir / f"bo_{label}.graded.json"), str(run_dir / "graded.json")], check=False)
        acc, overall = 100.0 * float(m.group(3)), 100.0 * float(m.group(2))
        value = self.value_of(acc, exec_loops)
        wall = time.time() - t0
        rec = {"thresholds": thr, "acc_pct": acc, "overall_unweighted_pct": overall, "exec_loops": exec_loops, "rel_compute_pct": 100.0 * exec_loops / 4.0,
               "tok_s_shards": sum(r["mean_tokens"] * r["n"] for r in results) / wall, "mean_tokens": sum(r["mean_tokens"] * r["n"] for r in results) / self.n_items,
               "trunc_rate": sum(r["trunc_rate"] * r["n"] for r in results) / self.n_items, "shards": k, "wall_s": wall, "gpu": gpu,
               "acc_floor_pct": self.acc_floor, "value": value, "status": "ok", "run_dir": str(run_dir)}
        self._eval_records.append(rec)
        self._save_item_cost(run_dir)
        json.dump(rec, open(run_dir / "record.json", "w"), indent=1)
        with self._lock:
            with open(self.work_dir / "evaluations.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"[bfcl-router] thr={['%.3f' % v for v in thr[:3]]} acc={acc:.1f}% (unweighted {overall:.1f}%) loops={exec_loops:.3f} value={value:.2f} "
              f"({wall / 60:.0f} min, {k} shards, gpu {gpu})", flush=True)
        return value, True

    def format_for_display(self, value: float) -> float:
        return value
