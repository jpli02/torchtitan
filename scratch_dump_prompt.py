import sys
sys.path.insert(0, "scripts")
from evaluate_humaneval_evalplus import _EVALPLUS_INSTRUCTION, _EVALPLUS_RESPONSE
from torchtitan.config import ConfigManager
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from evalplus.data import get_human_eval_plus

cfg = ConfigManager().parse_args(["--module", "ouro", "--config", "ouro_1_4b"])
tok = HuggingFaceTokenizer.Config().build(tokenizer_path=cfg.hf_assets_path)

prompt = get_human_eval_plus()["HumanEval/0"]["prompt"]

# Exactly mirror _generate_evalplus_chat defaults: system_prompt=None, prefill=True
user = f"{_EVALPLUS_INSTRUCTION}\n```\n{prompt.strip()}\n```\n"
messages = [{"role": "user", "content": user}]
prompt_str = tok.apply_chat_template(messages, add_generation_prompt=True)
prompt_str = prompt_str + f"{_EVALPLUS_RESPONSE}\n```python\n"

ids = tok.encode(prompt_str, add_bos=False, add_eos=False)  # chat_add_bos=False default

print("==================== RENDERED PROMPT (repr) ====================")
print(repr(prompt_str))
print("\n==================== RENDERED PROMPT (raw) ====================")
print(prompt_str)
print("==================== END ====================")
print(f"\n[chars={len(prompt_str)}  tokens={len(ids)}  add_bos=False]")
