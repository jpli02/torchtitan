#!/usr/bin/env python3
"""torchtitan Ouro diagnostic (+ best-effort HF/vLLM-style parity).

Primary, dependency-free signals:
  * gather early-exit counter -- how many tokens the old threshold-gather would
    have scored from a pre-final UT step at threshold=1.0 (the suspected bug).
  * torchtitan next-token predictions on a couple of prompts (sanity).

If the HF reference (modeling_ouro.py) loads under the installed transformers, it
also reports logit/argmax parity with the vLLM-style inference path: full R=4
recurrence and final-step logits.  The released HF config has
early_exit_threshold=1.0, which routes HF through threshold-gather logits; disable
that before comparing so this diagnostic matches vLLM's ouro.py.
"""
import sys, torch
import sys as _sys
sys.path.insert(0, "scripts")
from evaluate_humaneval import _load_model
from transformers import AutoTokenizer

HF_DIR = "./assets/hf/Ouro-1.4B"
PROMPTS = [
    "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
    "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n",
    "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    \"\"\"\n",
]

tok = AutoTokenizer.from_pretrained(HF_DIR, trust_remote_code=True)
all_ids = [tok(p, return_tensors="pt").input_ids for p in PROMPTS]

print("loading torchtitan OuroModel (R=4, threshold=1.0) ...")
tt, _ = _load_model("ouro", "ouro_1_4b", None, HF_DIR, early_exit_threshold=1.0)
tt.eval()
tt_dev = next(tt.parameters()).device

tt_logits = []
with torch.no_grad():
    for ids in all_ids:
        tt_logits.append(tt(ids.to(tt_dev)).float())
for p, lg in zip(PROMPTS, tt_logits):
    print(f"  TT next-token for {p[:36]!r}: {tok.decode(lg.argmax(-1)[0,-1].item())!r}")
print(f"\nGATHER early-exit tokens: {tt._gather_early_exits}/{tt._gather_total}  "
      f"(>0 => the old threshold-gather WAS scoring tokens from a pre-final UT step)")

# --- best-effort HF parity (may fail on transformers-version incompat) ---
print("\nattempting HF reference parity ...")
try:
    import transformers.modeling_rope_utils as _rope

    def _default_rope(config, device=None, seq_len=None, layer_type=None, **kw):
        params = getattr(config, "rope_parameters", None) or getattr(
            config, "rope_scaling", None
        ) or {}
        base = float(getattr(config, "rope_theta", 10000.0))
        if isinstance(params, dict):
            base = float(params.get("rope_theta", base))
        dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        inv = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float32
                )
                / dim
            )
        )
        return inv, 1.0

    if "default" not in _rope.ROPE_INIT_FUNCTIONS:
        _rope.ROPE_INIT_FUNCTIONS["default"] = _default_rope
    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = AutoConfig.from_pretrained(HF_DIR, trust_remote_code=True)
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = getattr(cfg, "eos_token_id", None) or 0
    # Transformers 5.8 initializes missing RoPE buffers by calling
    # module.compute_default_rope_parameters for remote-code RotaryEmbedding
    # classes. The released Ouro modeling file predates that method.
    hf_cls = get_class_from_dynamic_module(cfg.auto_map["AutoModelForCausalLM"], HF_DIR)
    modeling_mod = _sys.modules[hf_cls.__module__]
    if not hasattr(modeling_mod.OuroRotaryEmbedding, "compute_default_rope_parameters"):
        modeling_mod.OuroRotaryEmbedding.compute_default_rope_parameters = staticmethod(
            _default_rope
        )
    cfg.early_exit_threshold = None
    hf = AutoModelForCausalLM.from_pretrained(
        HF_DIR, config=cfg, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    hf.early_exit_threshold = None
    hf.config.early_exit_threshold = None
    with torch.no_grad():
        for p, ids, ttl in zip(PROMPTS, all_ids, tt_logits):
            hfl = hf(ids.cuda()).logits.float()
            T = min(hfl.shape[1], ttl.shape[1])
            d = (hfl[:, :T] - ttl[:, :T].cuda()).abs()
            agree = (hfl[:, :T].argmax(-1)[0] == ttl[:, :T].cuda().argmax(-1)[0]).float().mean().item()
            print(f"  {p[:36]!r}: max|Δ|={d.max():.4f} mean|Δ|={d.mean():.5f} argmax_agree={agree:.3f}")
except Exception as e:
    print(f"  HF reference unavailable ({type(e).__name__}: {e}).")
    print("  -> rely on the gather counter above + the eval re-run pass@1 delta.")
