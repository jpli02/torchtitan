"""Write a model directory the evaluation scripts can serve: the released base weights with a LoRA update folded in and,
optionally, a refitted exit gate; the serving model file (decode-time early exit, executed-loop counter) and the tokenizer are
copied from the serving directory.

  CUDA_VISIBLE_DEVICES=0 python randrouter/export.py --out DIR [--lora FILE] [--gate FILE] [--check_n 12]
With --check_n the teacher-forced fidelity of a few routers is printed before (LoRA applied on the fly, float32 update) and
after the merge (bfloat16 weights, as served): a merged update smaller than the weights' bfloat16 spacing would be lost.
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
import tf_eval  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--lora", default=None)
ap.add_argument("--lora_r", type=int, default=32)
ap.add_argument("--lora_alpha", type=float, default=64.0)
ap.add_argument("--gate", default=None)
ap.add_argument("--teacher", default="/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl")
ap.add_argument("--check_n", type=int, default=0)
ap.add_argument("--check_routers", default="full;0.3;iid:0.25,0.25,0.25,0.25")
ap.add_argument("--mem_gb", type=float, default=0.0)
ap.add_argument("--serve_dir", default=core.SERVE_DIR,
                help="directory the model file and tokenizer are copied from (the exit-lab one for a model trained with the calibrated fill)")
ap.add_argument("--fill_maps", default=None, help="calibrated fill for the fidelity check (the model was trained with it)")
args = ap.parse_args()

if args.mem_gb:
    torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
model, tok = core.load_model(core.SERVE_DIR)
routers = [tf_eval.parse_router(s) for s in args.check_routers.split(";")]
val = core.load_teacher(args.teacher, split="val")[:args.check_n] if args.check_n else []
FILL = core.load_fill_maps(args.fill_maps) if args.fill_maps else None
if args.gate:
    g = torch.load(args.gate, map_location="cpu")
    with torch.no_grad():
        model.model.early_exit_gate.weight.copy_(g["weight"].to(model.model.early_exit_gate.weight))
        model.model.early_exit_gate.bias.copy_(g["bias"].to(model.model.early_exit_gate.bias))
    print(f"gate replaced by {args.gate} ({g.get('variant')})", flush=True)
if args.lora:
    core.add_lora(model, args.lora_r, args.lora_alpha)
    core.load_lora_state(model, torch.load(args.lora, map_location="cpu"))
    if val:
        print("before the merge (update applied on the fly):", flush=True)
        tf_eval.evaluate(model, val, routers, tok.pad_token_id, log=lambda s: print(s, flush=True), fill_maps=FILL)
    n = core.merge_lora(model)
    print(f"merged {n} LoRA updates into the base weights", flush=True)
if val:
    print("as exported:", flush=True)
    tf_eval.evaluate(model, val, routers, tok.pad_token_id, log=lambda s: print(s, flush=True), fill_maps=FILL)

os.makedirs(args.out, exist_ok=True)
state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
with safe_open(os.path.join(core.BASE_DIR, "model.safetensors"), "pt") as f:
    base_keys = set(f.keys())
assert set(state) == base_keys, f"state keys differ from the released file: {sorted(set(state) ^ base_keys)[:5]}"
save_file(state, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
for fn in ("config.json", "configuration_ouro.py", "modeling_ouro.py", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
           "vocab.json", "merges.txt"):
    shutil.copy(os.path.join(args.serve_dir, fn), os.path.join(args.out, fn))
json.dump({"base": core.BASE_DIR, "serve_dir": args.serve_dir, "lora": args.lora, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "gate": args.gate},
          open(os.path.join(args.out, "randrouter_export.json"), "w"), indent=1)
print(f"EXPORTED {args.out}", flush=True)
