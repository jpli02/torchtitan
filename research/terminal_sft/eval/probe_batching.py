"""Why does the model never emit >1 command per turn?

Measures the model's ACTUAL next-token distribution at the batching decision
point -- after the first command object closes, the next token is either ','
(another command follows) or ']' (list ends) -- in a real terminus-2 context.

Discriminates:
  P(',') ~ 0.19  -> conditional matches the data; decode-side suppression
  P(',') ~ 0.00  -> the model learned "never batch" (data reweighting needed)
"""
import sys, torch
sys.path.insert(0, "/home/jli199/torchtitan/scripts")
from evaluate_humaneval_evalplus import _load_hf_generate_model

CKPT = "/home/jli199/boptim_scratch/ouro_c10k_clean"
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

# Force the model to the batching decision: one complete command object, then
# let it choose what comes next.
PREFIX = (
    '{\n  "analysis": "I need to inspect the project.",\n'
    '  "plan": "List files, then read the README, then check python.",\n'
    '  "commands": [\n'
    '    {"keystrokes": "ls -la\\n", "duration": 0.1}'
)


def next_token_dist(text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        _, hidden_list, _ = model.model(
            input_ids=ids, use_cache=True,
            cache_position=torch.arange(ids.shape[1], device=model.device))
        logits = model.lm_head(hidden_list[-1][:, -1:, :])[0, -1].float()
    return torch.softmax(logits, dim=-1)


chat = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                               add_generation_prompt=True, tokenize=False,
                               enable_thinking=False)
probs = next_token_dist(chat + PREFIX)

top = torch.topk(probs, 10)
print("prefix ends:", repr(PREFIX[-45:]))
print("\ntop next-token candidates at the batching decision:")
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  {p:9.6f}  {tok.decode([i])!r}")

comma_p = 0.0
bracket_p = 0.0
pl = probs.tolist()
for i, p in enumerate(pl):
    if p < 1e-7:
        continue
    s = tok.decode([i])
    if "," in s:
        comma_p += p
    if "]" in s:
        bracket_p += p

print(f"\nP(token containing ',') = {comma_p:.6f}   <- continue batching")
print(f"P(token containing ']') = {bracket_p:.6f}   <- end the command list")
print(f"ratio  ']' : ','        = {bracket_p / max(comma_p, 1e-12):.1f} : 1")
print("\nreference: data-implied P(',') at T=0.7 would be ~0.195")
if comma_p < 0.01:
    print("VERDICT: model learned 'never batch' -> training-side fix (reweight data)")
elif comma_p > 0.10:
    print("VERDICT: conditional retains batching -> decode-side / prompt-side fix")
else:
    print("VERDICT: intermediate -- batching is suppressed but not erased")
