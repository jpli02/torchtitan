"""Which signal knows that a token may leave early? For every answer token of the BFCL search items (teacher-forced on the model's
own full-depth answers, one forward pass per item) this records, at each loop s = 1..4:
  * the token the language-model head would emit after loop s (pred_s) and whether it equals the full-depth token (the target);
  * the head's confidence after loop s: top-1 probability and the margin to the second token;
  * the exit gate's cumulative exit probability P_s (what the released thresholds use).
A token can safely leave after loop s when pred_s already equals the target; the question is which signal ranks those tokens first.

  CUDA_VISIBLE_DEVICES=0 python boptim_lcb/bfcl_signal_dump.py [n_items] [out.jsonl]          (analysis: bfcl_signal_analyze.py)
"""
import glob
import json
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

CKPT = "/home/jli199/boptim_scratch/ouro_base_kvshare"
DATA = sys.argv[3] if len(sys.argv) > 3 else "/srv/scratch/jli199/bfcl_stage/bfcl_search.jsonl"
WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 575
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{WORK}/signal_probs.jsonl"
GEN_GLOB = sys.argv[4] if len(sys.argv) > 4 else f"{WORK}/q1.000-1.000-1.000_*/record.json"

rows = [json.loads(l) for l in open(DATA) if l.strip()][:N]
full_dir = None
for rec in sorted(glob.glob(GEN_GLOB)):
    if json.load(open(rec)).get("status") == "ok":
        full_dir = rec.rsplit("/", 1)[0]
gens = [json.loads(l) for l in open(f"{full_dir}/gen.jsonl") if l.strip() and '"summary"' not in l[:12]][:N]
assert len(gens) == len(rows) and all(g["id"] == r["id"] for g, r in zip(gens, rows)), "full-depth answers do not line up with the data"

tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
config = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
config.total_ut_steps = 4
model = AutoModelForCausalLM.from_pretrained(CKPT, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
eos = tok.eos_token_id

with open(OUT, "w") as out, torch.inference_mode():
    for k, (r, g) in enumerate(zip(rows, gens)):
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"].rstrip()}], tokenize=False, add_generation_prompt=True) + "["
        p_ids = tok(text)["input_ids"]
        a_ids = tok(g["gen"][1:] if g["gen"].startswith("[") else g["gen"], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids + a_ids], device="cuda")
        _, hidden_list, gate_list = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        lo = len(p_ids) - 1                      # position t produces token t+1: answer tokens come from positions lo .. T-1
        target = a_ids + [eos]                   # the full-depth continuation, token by token
        lam = [torch.sigmoid(x.float().squeeze(-1))[0, lo:] for x in gate_list]
        rem, acc, cum = torch.ones_like(lam[0]), torch.zeros_like(lam[0]), []
        for i in range(len(lam) - 1):
            acc = acc + lam[i] * rem
            rem = rem * (1.0 - lam[i])
            cum.append(acc)
        rec = {"id": r["id"], "category": r["category"], "n": len(target), "target": target, "P": torch.stack(cum, 1).cpu().tolist(), "pred": [], "conf": [], "margin": []}
        for h in hidden_list:
            pr = torch.softmax(model.lm_head(h[0, lo:]).float(), dim=-1)
            top = torch.topk(pr, 2, dim=-1)
            rec["pred"].append(top.indices[:, 0].cpu().tolist())
            rec["conf"].append([round(v, 4) for v in top.values[:, 0].cpu().tolist()])
            rec["margin"].append([round(v, 4) for v in (top.values[:, 0] - top.values[:, 1]).cpu().tolist()])
        rec["P"] = [[round(v, 4) for v in p] for p in rec["P"]]
        out.write(json.dumps(rec) + "\n")
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(rows)} items", flush=True)
print("wrote", OUT)
