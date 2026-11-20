"""The teacher-forced tables of the experiments page, in one run per system: every router on ALL validation answers, with the
decoder's cache and with an exact cache.

  CUDA_VISIBLE_DEVICES=9 python randrouter/collect_tables.py --name base [--lora F] [--fill_maps F] [--gate F] [--mem_gb 8.3]
Writes /home/jli199/boptim_scratch/randrouter/tables/<name>.json: {"system": {...}, "n_seqs", "n_tokens", "routers": {spec: ...}}.
With --gate_signal also the released gate's loop-by-loop statistics (gate_signal.py) of the same system.
"""
import argparse
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
import tf_eval  # noqa: E402

ROUTERS = ("full;0.7;0.5;0.3;0.25,0.3,0.3;0.245,0.3,0.3;0.2;0.25,1.0,1.0;0.245,1.0,1.0;0.2,0.4,0.7;0.245,0.45,0.7;"
           "rank:0.5,0.5,0.5;rank:0.7,0.5,0.5;rank:0.8,0.5,0.5;iid:0.25,0.25,0.25,0.25")

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--lora", default=None)
ap.add_argument("--fill_maps", default=None)
ap.add_argument("--gate", default=None)
ap.add_argument("--routers", default=ROUTERS)
ap.add_argument("--teacher", default="/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl")
ap.add_argument("--budget_tok", type=int, default=2500)
ap.add_argument("--mem_gb", type=float, default=0.0)
ap.add_argument("--gate_signal", action="store_true")
args = ap.parse_args()

out_dir = "/home/jli199/boptim_scratch/randrouter/tables"
os.makedirs(out_dir, exist_ok=True)
if args.mem_gb:
    torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
model, tok = core.load_model(core.SERVE_DIR)
if args.lora:
    core.add_lora(model, 32, 64.0)
    core.load_lora_state(model, torch.load(args.lora, map_location="cpu"))
if args.gate:
    g = torch.load(args.gate, map_location="cpu")
    with torch.no_grad():
        model.model.early_exit_gate.weight.copy_(g["weight"].to(model.model.early_exit_gate.weight))
        model.model.early_exit_gate.bias.copy_(g["bias"].to(model.model.early_exit_gate.bias))
fm = core.load_fill_maps(args.fill_maps) if args.fill_maps else None
seqs = core.load_teacher(args.teacher, split="val")
print(f"{args.name}: {len(seqs)} validation answers", flush=True)
res = tf_eval.evaluate(model, seqs, [tf_eval.parse_router(s) for s in args.routers.split(";")], tok.pad_token_id, args.budget_tok,
                       log=lambda s: print(s, flush=True), fill_maps=fm)
json.dump({"system": {"lora": args.lora, "fill_maps": args.fill_maps, "gate": args.gate}, "n_seqs": len(seqs),
           "n_tokens": sum(len(s["ids"]) - s["n_prompt"] for s in seqs), "routers": res}, open(f"{out_dir}/{args.name}.json", "w"), indent=1)
del model
torch.cuda.empty_cache()
if args.gate_signal:
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_signal.py"), "--split", "val",
           "--budget_tok", str(args.budget_tok), "--out", f"{out_dir}/{args.name}_gate_signal.json"]
    if args.lora:
        cmd += ["--lora", args.lora]
    if args.gate:
        cmd += ["--gate", args.gate]
    if args.mem_gb:
        cmd += ["--mem_gb", str(args.mem_gb)]
    subprocess.run(cmd, check=False)
print(f"WROTE {out_dir}/{args.name}.json", flush=True)
