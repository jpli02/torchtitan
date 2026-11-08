"""Per-token teacher-forced correctness dump for the token-accuracy distribution analysis.

Usage: python fit_eval_dump.py CKPT RENDERED.jsonl OUT.jsonl [GROUP=new] [CAP=12288]
Full-context pass per row; writes one JSON line per row: {"i", "task_turns", "tokens": [[piece, role, correct, pos], ...]}
restricted to supervised tokens, with role in command / prose / other (see fit_eval_spans.py)."""
import json
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt, data, out = sys.argv[1:4]
group = sys.argv[4] if len(sys.argv) > 4 else "new"
CAP = int(sys.argv[5]) if len(sys.argv) > 5 else 12288
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()


def spans_of(text, key):
    res = []
    for m in re.finditer(r'"%s"\s*:\s*"' % key, text):
        i = j = m.end()
        while j < len(text):
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == '"':
                break
            j += 1
        res.append((i, j))
    return res


t0 = time.time()
n = 0
with open(out, "w") as fo, torch.inference_mode():
    for line in open(data):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["group"] != group:
            continue
        ids, labels = row["ids"][:CAP], row["labels"][:CAP]
        pieces = [tok.decode([t]) for t in ids]
        offsets, pos = [], 0
        for p in pieces:
            offsets.append((pos, pos + len(p)))
            pos += len(p)
        text = "".join(pieces)
        cmd = spans_of(text, "keystrokes")
        prose = spans_of(text, "analysis") + spans_of(text, "plan")
        inp = torch.tensor(ids[:-1], device="cuda")[None]
        pred = model(input_ids=inp).logits[0].argmax(-1).tolist()
        recs = []
        for k in range(len(ids) - 1):
            if labels[k + 1] == -100:
                continue
            a, b = offsets[k + 1]
            mid = (a + b) / 2
            role = "command" if any(s <= mid < e for s, e in cmd) else ("prose" if any(s <= mid < e for s, e in prose) else "other")
            recs.append([pieces[k + 1], role, int(pred[k] == ids[k + 1]), k + 1, tok.decode([pred[k]]) if pred[k] != ids[k + 1] else ""])
        fo.write(json.dumps({"i": row["i"], "tokens": recs}) + "\n")
        n += 1
        if n % 20 == 0:
            print(f"  {n} rows, {time.time()-t0:.0f}s", flush=True)
print(f"dumped {n} rows -> {out} in {time.time()-t0:.0f}s")
