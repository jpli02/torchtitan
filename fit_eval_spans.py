"""Teacher-forced fit split by token role: command keystrokes vs the rest of the assistant turns.

Usage: python fit_eval_spans.py CKPT RENDERED.jsonl OUT.json [GROUP=new] [MAX_ROWS] [CAP=12288]
Full-context pass per row (like fit_eval_full.py). Each supervised token is classified by decoding the row's token
pieces and locating the "keystrokes": "..." string values of the JSON actions in the assistant text: tokens inside
those strings are COMMAND tokens, tokens inside "analysis"/"plan" strings are PROSE, the JSON scaffolding is OTHER.
Reports top-1 accuracy / CE per role, plus the share of turns whose command tokens are all predicted correctly (the
replay probability of a command under greedy decoding)."""
import json
import re
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt, data, out = sys.argv[1:4]
group = sys.argv[4] if len(sys.argv) > 4 else "new"
max_rows = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
CAP = int(sys.argv[6]) if len(sys.argv) > 6 else 12288

tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
agg = {"command": [0, 0.0, 0], "prose": [0, 0.0, 0], "other": [0, 0.0, 0]}
turn_stats = {"n_cmd_turns": 0, "cmd_turns_all_correct": 0, "cmd_tokens_per_turn": []}
t0 = time.time()
n_rows = 0


def spans_of(text, key):
    """Character spans of the string values of "<key>": "..." in JSON-ish text (handles escaped quotes)."""
    out = []
    for m in re.finditer(r'"%s"\s*:\s*"' % key, text):
        i = m.end()
        j = i
        while j < len(text):
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == '"':
                break
            j += 1
        out.append((i, j))
    return out


with torch.inference_mode():
    for line in open(data):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["group"] != group:
            continue
        if n_rows >= max_rows:
            break
        ids, labels = row["ids"][:CAP], row["labels"][:CAP]
        pieces = [tok.decode([t]) for t in ids]
        offsets, pos = [], 0
        for p in pieces:
            offsets.append((pos, pos + len(p)))
            pos += len(p)
        text = "".join(pieces)
        cmd = spans_of(text, "keystrokes")
        prose = spans_of(text, "analysis") + spans_of(text, "plan")

        def role(k):
            a, b = offsets[k]
            mid = (a + b) / 2
            if any(s <= mid < e for s, e in cmd):
                return "command"
            if any(s <= mid < e for s, e in prose):
                return "prose"
            return "other"

        inp = torch.tensor(ids[:-1], device="cuda")[None]
        tgt = torch.tensor(labels[1:], device="cuda")
        logits = model(input_ids=inp).logits[0]
        mask = tgt != -100
        pred = logits.argmax(-1)
        ce = F.cross_entropy(logits.float(), tgt.clamp(min=0), reduction="none")
        ok = (pred == tgt)
        del logits
        roles = [role(k + 1) for k in range(len(ids) - 1)]  # target token k+1
        for k in range(len(ids) - 1):
            if not bool(mask[k]):
                continue
            r = roles[k]
            agg[r][0] += int(ok[k])
            agg[r][1] += float(ce[k])
            agg[r][2] += 1
        # per command span: all tokens correct?
        for s, e in cmd:
            toks = [k for k in range(len(ids) - 1) if bool(mask[k]) and s <= (offsets[k + 1][0] + offsets[k + 1][1]) / 2 < e]
            if toks:
                turn_stats["n_cmd_turns"] += 1
                turn_stats["cmd_tokens_per_turn"].append(len(toks))
                turn_stats["cmd_turns_all_correct"] += int(all(bool(ok[k]) for k in toks))
        n_rows += 1
        if n_rows % 20 == 0:
            print(f"  {n_rows} rows, {time.time()-t0:.0f}s", flush=True)

summary = {r: {"n": n, "acc": c / n if n else None, "ce": s / n if n else None} for r, (c, s, n) in agg.items()}
summary["command_spans"] = turn_stats["n_cmd_turns"]
summary["command_span_all_correct_rate"] = turn_stats["cmd_turns_all_correct"] / max(turn_stats["n_cmd_turns"], 1)
summary["command_tokens_per_span_mean"] = sum(turn_stats["cmd_tokens_per_turn"]) / max(len(turn_stats["cmd_tokens_per_turn"]), 1)
json.dump({"ckpt": ckpt, "data": data, "group": group, "rows": n_rows, "summary": summary}, open(out, "w"), indent=1)
print(f"\n{ckpt.split('/')[-1]} on {group} rows (n={n_rows}) {time.time()-t0:.0f}s")
for r in ("command", "prose", "other"):
    s = summary[r]
    if s["n"]:
        print(f"  {r:8s} tokens={s['n']:7d} acc={s['acc']:.4f} ce={s['ce']:.3f}")
print(f"  command spans={summary['command_spans']} all-correct rate={summary['command_span_all_correct_rate']:.3f} tokens/span={summary['command_tokens_per_span_mean']:.1f}")
