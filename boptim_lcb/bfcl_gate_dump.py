"""Distribution of the exit gate's cumulative exit probabilities over the decoded tokens of BFCL answers (full depth, teacher-forced
on the model's own full-depth answers). For every answer token it records P1, P2, P3 = cumulative exit probability after loops
1, 2, 3 (stick-breaking over the gate's sigmoid outputs; loop 4 always exits), i.e. the numbers the exit thresholds are compared with.
A token exits after loop s under the router (q1, q2, q3) when P_s >= q_s for the first time.

  CUDA_VISIBLE_DEVICES=0 python boptim_lcb/bfcl_gate_dump.py [n_items] [out.jsonl]          (one forward pass per item)
"""
import glob
import json
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

CKPT = "/home/jli199/boptim_scratch/ouro_base_kvshare"
DATA = "/srv/scratch/jli199/bfcl_stage/bfcl_search.jsonl"
WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 575
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{WORK}/gate_probs.jsonl"

rows = [json.loads(l) for l in open(DATA) if l.strip()][:N]
full_dir = None
for rec in sorted(glob.glob(f"{WORK}/q1.000-1.000-1.000_*/record.json")):
    if json.load(open(rec)).get("status") == "ok":
        full_dir = rec.rsplit("/", 1)[0]
gens = [json.loads(l) for l in open(f"{full_dir}/gen.jsonl") if l.strip() and '"summary"' not in l[:12]][:N]
assert len(gens) == len(rows) and all(g["id"] == r["id"] for g, r in zip(gens, rows)), "full-depth answers do not line up with the data"

tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
config = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
config.total_ut_steps = 4
model = AutoModelForCausalLM.from_pretrained(CKPT, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

with open(OUT, "w") as out, torch.inference_mode():
    for k, (r, g) in enumerate(zip(rows, gens)):
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"].rstrip()}], tokenize=False, add_generation_prompt=True) + "["
        p_ids = tok(text)["input_ids"]
        a_ids = tok(g["gen"][1:], add_special_tokens=False)["input_ids"] if g["gen"].startswith("[") else tok(g["gen"], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids + a_ids], device="cuda")
        _, _, gate_list = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        lam = [torch.sigmoid(x.float().squeeze(-1))[0] for x in gate_list]          # per loop: [T]
        rem = torch.ones_like(lam[0])
        cum, acc = [], torch.zeros_like(lam[0])
        for i in range(len(lam) - 1):
            acc = acc + lam[i] * rem
            rem = rem * (1.0 - lam[i])
            cum.append(acc)
        # position t decides the depth used to produce token t+1: the answer tokens are produced at positions len(p)-1 .. T-1
        lo = len(p_ids) - 1
        P = torch.stack(cum, dim=1)[lo:].cpu().tolist()
        toks = tok.convert_ids_to_tokens(a_ids) + ["<eos>"]
        out.write(json.dumps({"id": r["id"], "category": r["category"], "n": len(P), "P": [[round(v, 4) for v in p] for p in P], "next_tok": toks[:len(P)]}) + "\n")
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(rows)} items", flush=True)
print("wrote", OUT)
