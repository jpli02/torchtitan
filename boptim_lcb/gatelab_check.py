"""Check of the gate-lab serving path: the released gate, saved as a router file and loaded through decode_router, must decode
exactly like the released serving model (same answers, same executed loops). Also writes a router trained from scratch with
the Stage-II loss and decodes with it, to see that a non-linear router runs.

  CUDA_VISIBLE_DEVICES=9 python boptim_lcb/gatelab_check.py [--n 6]
"""
import argparse
import json
import os
import subprocess
import sys

WT = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft"
PY = "/home/jli199/boptim-agent/.venv/bin/python"
OUT = "/home/jli199/boptim_scratch/randrouter/gatelab"
SORTED = "/home/jli199/boptim_scratch/randrouter/real/base_fill24_q0.25-0.3-0.3_v2/data_sorted.jsonl"   # 342 test problems, shortest prompts first

sys.path.insert(0, os.path.join(WT, "randrouter"))


def run(ckpt, data, n, out, cfgs, thr):
    cmd = [PY, os.path.join(WT, "depth_eval.py"), ckpt, data, "--task", "lcb", "--steps", "4", "--n", str(n), "--seed_offset", "0",
           "--batch", "1", "--max_new", "1024", "--assistant_prefix", "```python\\n", "--cfg", "decode_early_exit_threshold=" + json.dumps(thr),
           "--out", out]
    for c in cfgs:
        cmd += ["--cfg", c]
    log = subprocess.run(cmd, cwd=WT, capture_output=True, text=True, env=dict(os.environ, PYTORCH_ALLOC_CONF="expandable_segments:True")).stdout
    res = None
    for line in log.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    rows = [json.loads(l) for l in open(out) if l.strip() and "question_id" in l] if os.path.exists(out) else []
    return res, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    import nas_fit

    os.makedirs(OUT, exist_ok=True)
    data = os.path.join(OUT, f"lcb_short{a.n}.jsonl")
    with open(SORTED) as f, open(data, "w") as g:
        for i, line in enumerate(f):
            if i >= a.n:
                break
            g.write(line)
    rel = os.path.join(OUT, "router_released_linear.pt")
    nas_fit.save_router(rel, nas_fit.released_gate_router(), {"layers": 0}, note="the released gate")
    thr = [0.3, 0.3, 0.3, 1.0]
    r0, rows0 = run("/home/jli199/boptim_scratch/ouro_base_kvshare", data, a.n, os.path.join(OUT, "chk_kvshare.jsonl"), [], thr)
    r1, rows1 = run("/home/jli199/boptim_scratch/ouro_base_gatelab", data, a.n, os.path.join(OUT, "chk_gatelab_released.jsonl"), [f"decode_router={rel}"], thr)
    same = sum(x.get("output") == y.get("output") for x, y in zip(rows0, rows1))
    print(f"released serving model: exec loops {r0 and r0.get('exec_loops')}, {r0 and r0.get('mean_tokens')} tokens")
    print(f"gate lab + released gate as router file: exec loops {r1 and r1.get('exec_loops')}, {r1 and r1.get('mean_tokens')} tokens")
    print(f"identical answers: {same} of {len(rows0)}")
    # a router trained from scratch (MLP, LayerNorm), Stage-II loss on the cached training states
    feats = nas_fit.load_feats("/home/jli199/boptim_scratch/randrouter/nas_feats_train.pt")
    spec = {"layers": 1, "hidden": 128, "act": "gelu", "norm": True}
    router, diag = nas_fit.fit_router(feats, spec, 0.005)
    mlp = os.path.join(OUT, "router_check_L1H128gelu_ln.pt")
    nas_fit.save_router(mlp, router, spec, gamma=0.005, fit=diag)
    r2, rows2 = run("/home/jli199/boptim_scratch/ouro_base_gatelab", data, a.n, os.path.join(OUT, "chk_gatelab_mlp.jsonl"), [f"decode_router={mlp}"], [0.5, 0.5, 0.5, 1.0])
    print(f"gate lab + MLP router trained from scratch (gamma 0.005, threshold 0.5): exec loops {r2 and r2.get('exec_loops')}, "
          f"{r2 and r2.get('mean_tokens')} tokens, {len(rows2)} answers; fit {diag}")
    print("GATELAB CHECK done")


if __name__ == "__main__":
    main()
