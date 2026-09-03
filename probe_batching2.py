"""Corrected batching probe.

The first probe mis-aggregated: the top token ']," is a SINGLE token containing
both ']' and ',' -- it CLOSES the list (the comma precedes "task_complete").
Counting it as evidence of batching was wrong.

Correct question: what is P(the model starts ANOTHER command object)? That means
following the separator branches and asking whether '{' comes next. Resolve it
by two-step lookahead instead of guessing from single-token strings.
"""
import os, sys, torch
sys.path.insert(0, "/home/jli199/torchtitan/scripts")
from evaluate_humaneval_evalplus import _load_hf_generate_model

CKPT = os.environ.get("OURO_PROBE_DIR", "/home/jli199/boptim_scratch/ouro_c10k_clean")
model, tok, eos_ids = _load_hf_generate_model(CKPT, dtype=torch.bfloat16, attn_impl="sdpa")
model.eval()

PROMPT = (
    "You are an AI assistant tasked with solving command-line tasks in a Linux "
    "environment. Format your response as JSON with the following structure:\n"
    '{\n  "analysis": "...",\n  "plan": "...",\n'
    '  "commands": [{"keystrokes": "ls -la\\n", "duration": 0.1}],\n'
    '  "task_complete": false\n}\n\n'
    "Task Description:\nInspect the project: list files, show the README, and "
    "print the python version.\n\n"
    "Current terminal state:\nCurrent Terminal Screen:\nroot@abc:/app#"
)
PREFIX = ('{\n  "analysis": "I need to inspect the project.",\n'
          '  "plan": "List files, then read the README, then check python.",\n'
          '  "commands": [\n'
          '    {"keystrokes": "ls -la\\n", "duration": 0.1}')


def dist(text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        _, hidden, _ = model.model(
            input_ids=ids, use_cache=True,
            cache_position=torch.arange(ids.shape[1], device=model.device))
        return torch.softmax(model.lm_head(hidden[-1][:, -1:, :])[0, -1].float(), dim=-1)


chat = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                               add_generation_prompt=True, tokenize=False,
                               enable_thinking=False)
base = chat + PREFIX
p1 = dist(base)
top = torch.topk(p1, 8)

p_new_cmd = 0.0   # mass that leads to another command object
p_close = 0.0     # mass that closes the list
print("step-1 candidates, resolved by one-step lookahead where ambiguous:")
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    s = tok.decode([i])
    if "]" in s:
        p_close += p
        print(f"  {p:9.6f}  {s!r:12s} -> CLOSES list")
        continue
    # ambiguous separator: look one token ahead
    p2 = dist(base + s)
    t2 = torch.topk(p2, 5)
    nxt = [(pp, tok.decode([ii])) for pp, ii in zip(t2.values.tolist(), t2.indices.tolist())]
    brace = sum(pp for pp, ss in nxt if "{" in ss)
    close = sum(pp for pp, ss in nxt if "]" in ss)
    lead = "NEW CMD" if brace > close else ("CLOSES" if close > brace else "unclear")
    p_new_cmd += p * brace
    p_close += p * close
    print(f"  {p:9.6f}  {s!r:12s} -> then {[(round(a,3), b) for a, b in nxt[:3]]}  => {lead}")

print(f"\nP(start another command) ~= {p_new_cmd:.5f}")
print(f"P(close command list)    ~= {p_close:.5f}")
print(f"reference: data-implied P(batch) at T=0.7 would be ~0.195")
if p_new_cmd < 0.02:
    print("VERDICT: batching effectively erased -> training-side fix (reweight data)")
elif p_new_cmd > 0.10:
    print("VERDICT: batching retained -> decode/prompt-side fix")
else:
    print("VERDICT: strongly suppressed but non-zero")
