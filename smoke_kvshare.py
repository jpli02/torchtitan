"""Smoke for SharedKVCache: (1) prefill logits identical to the stock cache, (2) first decode
token identical, (3) greedy generations per mode, (4) cache bytes per generated token.
Usage: python smoke_kvshare.py CKPT_DIR   (a *_kvshare dir)
"""
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ckpt = sys.argv[1]
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
qs = ["Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
      "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"]
texts = [tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                                 tokenize=False, add_generation_prompt=True) for q in qs]
enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
modes = {"stock": {}, "share_decode": {"kv_share_decode": True}, "share+collapse": {"kv_share_decode": True, "kv_collapse_prefill": True}}
ref_logits = ref_first = None
outs = {}
for name, cfg in modes.items():
    config = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    for k, v in cfg.items():
        setattr(config, k, v)
    model = AutoModelForCausalLM.from_pretrained(ckpt, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    with torch.inference_mode():
        o = model(**enc, use_cache=True)
        logits = o.logits[:, -1, :].float()
        cache = o.past_key_values
        first = logits.argmax(-1)
        if ref_logits is None:
            ref_logits, ref_first = logits, first
            print(f"[{name}] cache type {type(cache).__name__}")
        else:
            print(f"[{name}] cache type {type(cache).__name__}; prefill last-logit max|diff| vs stock = {(logits - ref_logits).abs().max().item():.3e}; "
                  f"first token equal: {bool((first == ref_first).all())}")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        gen = model.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=tok.pad_token_id, return_dict_in_generate=True)
        dt = time.time() - t0
        seqs = gen.sequences[:, enc["input_ids"].shape[1]:]
        cache = getattr(gen, "past_key_values", None)
        cb = cache.cache_bytes() if cache is not None and hasattr(cache, "cache_bytes") else None
        if cb is None and cache is not None and hasattr(cache, "key_cache"):
            cb = sum(t.numel() * t.element_size() for t in cache.key_cache + cache.value_cache if t is not None)
        ntok = int((seqs != tok.pad_token_id).sum().item())
        outs[name] = tok.batch_decode(seqs, skip_special_tokens=True)
        print(f"[{name}] 2x96 greedy: {dt:.1f}s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
              f"cache {cb/2**20 if cb else float('nan'):.0f} MiB for {enc['input_ids'].shape[1]} prompt + {ntok} gen tokens")
        for i, o_ in enumerate(outs[name]):
            print(f"   [{i}] {o_[:110]!r}")
    del model
    torch.cuda.empty_cache()
for name in ("share_decode", "share+collapse"):
    same = [outs[name][i] == outs["stock"][i] for i in range(len(qs))]
    print(f"{name}: generations identical to stock: {same}")
