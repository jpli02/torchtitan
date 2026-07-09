#!/usr/bin/env python3
"""Generate HumanEval solutions for the Ouro model in EvalPlus sample format.

This reproduces the paper's HumanEval protocol (Table 7): the *base* (pre-SFT)
model is run as RAW CODE COMPLETION (no chat template), at full R=4 recurrence,
and scored with the EvalPlus framework (which sanitizes generations and runs the
HumanEval + HumanEval+ test suites). lm-eval-harness's `humaneval` task scores a
few points lower because it does no sanitization; use this for the paper number.

This script only *generates* (one disjoint shard per GPU when --num_shards>1) and
writes a JSONL of EvalPlus samples ({"task_id", "solution"}); the surrounding
slurm script concatenates the shards and runs `evalplus.sanitize` +
`evalplus.evaluate` once on the combined file.

Model loading and the greedy no-KV-cache decode loop are reused from
evaluate_humaneval.py so the two paths stay byte-for-byte identical on
tokenization, recurrence, and stopping.

Usage:
    python scripts/evaluate_humaneval_evalplus.py \
        --module ouro --config ouro_1_4b \
        --hf_checkpoint ./assets/hf/Ouro-1.4B \
        --early_exit_threshold 1.0 --max_gen_toks 1024 \
        --num_shards 2 --shard_index 0 \
        --output_path outputs/he_evalplus/shard0.jsonl
"""

import argparse
import json
import pathlib
import time
from types import SimpleNamespace

import torch

# Same-dir import: reuse the exact model loader + lm wrapper / greedy loop.
from evaluate_humaneval import OuroLM, _load_model
from torchtitan.tools.logging import init_logger, logger

# Standard HumanEval completion stop strings (matches lm-eval's humaneval task).
# EvalPlus's sanitize() does the final extraction of the target function, so these
# only need to keep the model from running into an obviously-new top-level block.
HUMANEVAL_STOP = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]


def _score_samples_evalplus(samples_path):
    """EvalPlus base pass@1 over a samples.jsonl: evalplus.sanitize (AST
    extraction) then exec each sanitized solution against the base HumanEval test
    with a per-problem timeout. Returns (n_pass, n_total). Subset-safe (does not
    call evalplus.evaluate, which asserts all 164 problems are present)."""
    import subprocess, sys, signal
    from evalplus.data import get_human_eval_plus
    prob = get_human_eval_plus()
    samples_path = pathlib.Path(samples_path)
    n_total = sum(1 for l in open(samples_path) if l.strip())
    subprocess.run([sys.executable, "-m", "evalplus.sanitize", "--samples",
                    str(samples_path)], check=True)
    san = samples_path.with_name(samples_path.stem + "-sanitized.jsonl")
    by_tid = {json.loads(l)["task_id"]: json.loads(l)["solution"]
              for l in open(san) if l.strip()}
    class _TO(Exception):
        pass
    signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_TO()))
    npass = 0
    for tid, sol in by_tid.items():
        p = prob[tid]
        src = sol + "\n" + p["test"] + f"\ncheck({p['entry_point']})\n"
        signal.alarm(10)
        try:
            exec(compile(src, "<s>", "exec"), {}); npass += 1
        except Exception:
            pass
        finally:
            signal.alarm(0)
    return npass, n_total


def _generate_completion(lm: OuroLM, prompt: str, max_new: int) -> str:
    """Raw-completion greedy decode for a single HumanEval prompt (base protocol)."""
    input_ids = torch.tensor(
        lm._tokenizer.encode(prompt, add_bos=lm._add_bos, add_eos=False),
        dtype=torch.long,
        device=lm._device,
    ).unsqueeze(0)
    if input_ids.shape[1] > lm._max_length - max_new:
        input_ids = input_ids[:, -(lm._max_length - max_new):]

    stop_ids = [
        lm._tokenizer.encode(s, add_bos=False, add_eos=False) for s in HUMANEVAL_STOP
    ]
    gen_ids = lm._greedy_until(input_ids, stop_ids, max_new)
    text = lm._tokenizer.decode(gen_ids[0].tolist())
    # Trim at the earliest stop string (sanitize will still tidy the rest).
    for s in HUMANEVAL_STOP:
        if s in text:
            text = text[: text.index(s)]
    return text


def _generate_chat_completion(
    lm: OuroLM, prompt: str, entry_point: str, max_new: int
) -> str:
    """Chat-protocol greedy decode for one HumanEval prompt.

    Mirrors ``OuroLM.generate_until``'s chat branch exactly (ChatML user turn via
    ``_chat_user_content`` + ``apply_chat_template``, stop at ``<|im_end|>``/EOS,
    then ``_extract_chat_body`` to recover the indented body), so this stays
    byte-for-byte identical to the lm-eval chat path. Returns the function body
    that continues ``prompt`` (the caller writes ``prompt + body``).
    """
    prompt_str = lm._tokenizer.apply_chat_template(
        [{"role": "user", "content": lm._chat_user_content(prompt)}],
        add_generation_prompt=True,
    )
    input_ids = torch.tensor(
        lm._tokenizer.encode(prompt_str, add_bos=lm._add_bos, add_eos=False),
        dtype=torch.long,
        device=lm._device,
    ).unsqueeze(0)
    if input_ids.shape[1] > lm._max_length - max_new:
        input_ids = input_ids[:, -(lm._max_length - max_new):]

    extra_stop = {lm._im_end_id} if lm._im_end_id is not None else set()
    gen_ids = lm._greedy_until(input_ids, [], max_new, extra_stop_token_ids=extra_stop)
    text = lm._tokenizer.decode(gen_ids[0].tolist())
    # _extract_chat_body reads req.doc["entry_point"]; shim a minimal req.
    req = SimpleNamespace(doc={"entry_point": entry_point})
    return lm._extract_chat_body(text, req)


# EvalPlus's canonical instruct prompt (provider/utility.py::make_raw_chat_prompt,
# codegen.py prefixes) -- the protocol the paper used for the EvalPlus code number.
_EVALPLUS_INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
_EVALPLUS_RESPONSE = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)


def _build_evalplus_prompt(render_chat, prompt, system_prompt, prefill, no_system):
    """Build the EvalPlus-canonical prompt string (shared by the no-cache
    torchtitan path and the KV-cache HF path so they stay byte-identical).

    ``render_chat`` maps a ChatML message list -> rendered string (each caller
    supplies its own tokenizer's apply_chat_template, since the tt and HF
    tokenizers differ in the ``tokenize`` default). See _generate_evalplus_chat
    for the system_prompt / prefill / no_system lever semantics.
    """
    user = f"{_EVALPLUS_INSTRUCTION}\n```\n{prompt.strip()}\n```\n"
    if no_system:
        # Template always forces a system turn; emit ChatML by hand without one.
        prompt_str = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    else:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user})
        prompt_str = render_chat(messages)
    if prefill:
        prompt_str = prompt_str + f"{_EVALPLUS_RESPONSE}\n```python\n"
    return prompt_str


def _generate_evalplus_chat(
    lm: OuroLM,
    prompt: str,
    max_new: int,
    system_prompt: str | None = None,
    prefill: bool = True,
    no_system: bool = False,
) -> str:
    """Generate a full self-contained solution using EvalPlus's canonical prompt.

    Replicates ``evalplus.provider.utility.make_raw_chat_prompt``: a user turn
    asking for a self-contained script (the problem fenced in ```), with the
    assistant turn *prefilled* by EvalPlus's response prefix and an opening
    ```python fence so the model continues inside it. (We append the prefill
    after ``add_generation_prompt`` instead of EvalPlus's magic-splitter trick;
    the resulting token stream is identical.) Returns the full script -- the
    caller submits it verbatim and EvalPlus sanitize extracts the entry function.

    Prompt-sweep levers for closing the gap to the paper's 0.744:
    - ``system_prompt``: None keeps the tokenizer template's silently-injected
      default ("You are a helpful assistant."); pass "" to suppress it (empty
      system turn) or any string to set a code-specific system message.
    - ``prefill``: False drops EvalPlus's assistant response prefix + ```python
      fence, letting the model open its own fence (sanitize still extracts).
    - ``no_system``: True removes the system turn *entirely* (no ``<|im_start|>
      system`` block at all). The tokenizer's ChatML template unconditionally
      injects a default system turn, so we bypass it and build the ChatML string
      by hand (user turn + assistant open) to match the template minus the system
      block. This is the paper author's suggested lever for the base model.
    """
    prompt_str = _build_evalplus_prompt(
        lambda m: lm._tokenizer.apply_chat_template(m, add_generation_prompt=True),
        prompt, system_prompt, prefill, no_system,
    )
    input_ids = torch.tensor(
        lm._tokenizer.encode(prompt_str, add_bos=lm._add_bos, add_eos=False),
        dtype=torch.long, device=lm._device,
    ).unsqueeze(0)
    if input_ids.shape[1] > lm._max_length - max_new:
        input_ids = input_ids[:, -(lm._max_length - max_new):]
    extra_stop = {lm._im_end_id} if lm._im_end_id is not None else set()
    gen_ids = lm._greedy_until(input_ids, [], max_new, extra_stop_token_ids=extra_stop)
    text = lm._tokenizer.decode(gen_ids[0].tolist())
    if prefill:
        # The model continues inside the opened ```python fence; cut at its close.
        if "```" in text:
            text = text[: text.index("```")]
    # Without prefill the model emits its own markdown; hand the full turn to
    # evalplus.sanitize, which extracts the fenced code / entry function.
    return text


# ---------------------------------------------------------------------------
# KV-cache path: reuse the official HF modeling_ouro.py (UniversalTransformerCache)
# for O(n) incremental decoding instead of the O(n^2) no-cache torchtitan loop.
# Parity (scripts/parity_hf_vs_tt.py) confirmed the HF forward matches the
# torchtitan decode's argmax exactly, so outputs stay comparable -- only faster.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _ouro_exit_steps(gate_list, threshold: float) -> torch.Tensor:
    """Per-token UT exit step from the gate stick-breaking PDF (0-based; loops
    used == step+1). Replicates modeling_ouro's threshold-gather and torchtitan's
    _adaptive_forward: first step whose cumulative exit prob >= threshold, else the
    last step. gate_list[i] is [B, T, 1]; returns [B, T]."""
    R = len(gate_list)
    remaining = torch.ones_like(gate_list[0].squeeze(-1), dtype=torch.float32)
    cumulative = torch.zeros_like(remaining)
    exit_steps = torch.full(remaining.shape, R - 1, dtype=torch.long,
                            device=remaining.device)
    exited = torch.zeros_like(remaining, dtype=torch.bool)
    for i, g in enumerate(gate_list):
        lam = torch.sigmoid(g.squeeze(-1).float())
        p_i = remaining if i == R - 1 else lam * remaining
        remaining = remaining * (1.0 - lam)
        cumulative = cumulative + p_i
        newly = (~exited) & ((cumulative >= threshold) | (i == R - 1))
        exit_steps = torch.where(newly, torch.full_like(exit_steps, i), exit_steps)
        exited = exited | newly
    return exit_steps


def _load_hf_generate_model(hf_dir: str, dtype: torch.dtype):
    """Load the released HF Ouro model for cached generation. Mirrors the RoPE /
    transformers-version shims from parity_hf_vs_tt.py. Generation drives the
    OuroModel directly (full-depth KV cache) and applies the early-exit gate to
    pick each token's output step (see _generate_evalplus_chat_hf), so the config
    threshold here is irrelevant."""
    import sys as _sys
    import transformers.modeling_rope_utils as _rope
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    def _default_rope(config, device=None, seq_len=None, layer_type=None, **kw):
        params = (getattr(config, "rope_parameters", None)
                  or getattr(config, "rope_scaling", None) or {})
        base = float(getattr(config, "rope_theta", 10000.0))
        if isinstance(params, dict):
            base = float(params.get("rope_theta", base))
        dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads)
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(
            device=device, dtype=torch.float32) / dim))
        return inv, 1.0

    if "default" not in _rope.ROPE_INIT_FUNCTIONS:
        _rope.ROPE_INIT_FUNCTIONS["default"] = _default_rope
    cfg = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = getattr(cfg, "eos_token_id", None) or 0
    hf_cls = get_class_from_dynamic_module(cfg.auto_map["AutoModelForCausalLM"], hf_dir)
    modeling_mod = _sys.modules[hf_cls.__module__]
    if not hasattr(modeling_mod.OuroRotaryEmbedding, "compute_default_rope_parameters"):
        modeling_mod.OuroRotaryEmbedding.compute_default_rope_parameters = staticmethod(
            _default_rope)
    cfg.early_exit_threshold = None  # full R=4 recurrence (no adaptive exit)
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir, config=cfg, trust_remote_code=True, dtype=dtype).cuda().eval()
    model.early_exit_threshold = None
    model.config.early_exit_threshold = None
    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = {im_end_id}
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    logger.info(f"HF KV-cache model loaded (dtype={dtype}, eos_ids={sorted(eos_ids)})")
    return model, tok, eos_ids


@torch.no_grad()
def _generate_evalplus_chat_hf(hf_model, hf_tok, eos_ids, prompt, max_new,
                               system_prompt=None, prefill=True, no_system=False,
                               exit_threshold=None):
    """KV-cached greedy decode of one EvalPlus-canonical prompt via the HF model,
    WITH adaptive early exit.

    Drives OuroModel directly (not OuroForCausalLM) so we get per-UT-step hidden
    states + gate outputs. The cache is full-depth (every layer x UT step), so
    attention is exact under incremental decoding; the early-exit gate only
    selects which UT step's hidden feeds the logits for each token -- identical to
    the no-cache _adaptive_forward, but O(n) via the cache. exit_threshold<1.0
    enables early exit; >=1.0 or None => full R=4. Returns (text, loop_sum,
    loop_count) so the caller can aggregate avg_loops."""
    prompt_str = _build_evalplus_prompt(
        lambda m: hf_tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False),
        prompt, system_prompt, prefill, no_system,
    )
    dev = hf_model.device
    enc = hf_tok(prompt_str, return_tensors="pt", add_special_tokens=False)
    cur = enc.input_ids.to(dev)
    L = cur.shape[1]
    cache_position = torch.arange(L, device=dev)
    cache = None
    new_ids = []
    loop_sum = 0.0
    loop_count = 0
    thr = (float(exit_threshold)
           if (exit_threshold is not None and float(exit_threshold) < 1.0) else None)
    for step in range(max_new):
        base_out, hidden_list, gate_list = hf_model.model(
            input_ids=cur, past_key_values=cache, use_cache=True,
            cache_position=cache_position)
        cache = base_out.past_key_values
        R = len(gate_list)
        # exit step of the token that generates next (the last position)
        if thr is not None:
            es_last = int(_ouro_exit_steps(gate_list, thr)[0, -1].item())
        else:
            es_last = R - 1  # full recurrence
        logits = hf_model.lm_head(hidden_list[es_last][:, -1:, :])
        nt = int(logits[0, -1].argmax(-1))
        loop_sum += es_last + 1
        loop_count += 1
        if nt in eos_ids:
            break
        new_ids.append(nt)
        cur = torch.tensor([[nt]], device=dev)
        cache_position = torch.tensor([L + step], device=dev)
    text = hf_tok.decode(new_ids, skip_special_tokens=True)
    if prefill and "```" in text:
        text = text[: text.index("```")]
    return text, loop_sum, loop_count


def main():
    p = argparse.ArgumentParser(description="Generate EvalPlus HumanEval samples for Ouro")
    p.add_argument("--module", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", help="DCP checkpoint path")
    p.add_argument("--hf_checkpoint", help="HF safetensors checkpoint directory")
    p.add_argument("--max_gen_toks", type=int, default=1024)
    p.add_argument(
        "--early_exit_threshold", type=float, default=1.0,
        help="1.0 = full R=4 recurrence (paper base-model setting).",
    )
    p.add_argument("--early_exit_step", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Cap #problems (quick check).")
    p.add_argument(
        "--task_ids_file", default=None,
        help="File with one HumanEval task_id per line; restrict generation to "
        "these (applied before sharding). For targeted re-tests, e.g. only the "
        "problems that failed a prior run.",
    )
    p.add_argument(
        "--chat", action="store_true",
        help="Use the ChatML chat protocol (SFT-matched: apply_chat_template + "
        "body extraction) instead of raw completion. Required for the released "
        "instruct-style Ouro-1.4B, which echoes the prompt under raw completion.",
    )
    p.add_argument(
        "--evalplus_prompt", action="store_true",
        help="Use EvalPlus's canonical instruct prompt (self-contained-script "
        "instruction + assistant prefill), matching the paper's EvalPlus protocol. "
        "Implies chat formatting; the full generated script is submitted as-is.",
    )
    p.add_argument(
        "--fp32", action="store_true",
        help="Run the model forward in float32 (vs default bf16) to test whether "
        "precision is part of the gap to the paper.",
    )
    p.add_argument(
        "--system_prompt", type=str, default=None,
        help="(evalplus_prompt only) System message. Omit to keep the chat "
        "template's default 'You are a helpful assistant.'; pass '' to suppress "
        "it, or a string to set a code-specific system prompt.",
    )
    p.add_argument(
        "--no_prefill", action="store_true",
        help="(evalplus_prompt only) Drop EvalPlus's assistant response prefix + "
        "```python prefill; let the model open its own fence.",
    )
    p.add_argument(
        "--no_system", action="store_true",
        help="(evalplus_prompt only) Remove the system turn ENTIRELY (no "
        "<|im_start|>system block). The ChatML template always injects a default "
        "system prompt; this bypasses it. Author-suggested lever for the base "
        "model on HumanEval. Overrides --system_prompt.",
    )
    p.add_argument(
        "--kv_cache", action="store_true",
        help="(evalplus_prompt only) Generate via the HF modeling_ouro.py + "
        "UniversalTransformerCache for O(n) KV-cached decoding instead of the "
        "O(n^2) no-cache torchtitan loop (~10x faster; argmax-equivalent per "
        "parity). Requires --hf_checkpoint.",
    )
    p.add_argument("--output_path", default="outputs/he_evalplus/samples.jsonl")
    p.add_argument(
        "--evalplus_score", action="store_true",
        help="Compute EvalPlus base pass@1 (sanitize + base-test exec) over the "
        "generated samples and record it in --eval_summary_json.",
    )
    p.add_argument(
        "--eval_summary_json", default=None,
        help="Write an ouro_eval_summary JSON (pass@1, avg_loops, loop_sum/count, "
        "n_problems, early_exit_threshold) here — consumed by the boptim objective "
        "so it can tune the gate at the EvalPlus-canonical protocol.",
    )
    args = p.parse_args()

    if (args.checkpoint is None) == (args.hf_checkpoint is None):
        p.error("Provide exactly one of --checkpoint or --hf_checkpoint.")
    if args.kv_cache and not args.evalplus_prompt:
        p.error("--kv_cache is only implemented for --evalplus_prompt.")
    if args.kv_cache and args.hf_checkpoint is None:
        p.error("--kv_cache requires --hf_checkpoint (loads HF modeling_ouro.py).")

    init_logger()

    try:
        from evalplus.data import get_human_eval_plus
    except ImportError:
        raise SystemExit(
            "EvalPlus is not installed.\n  Run:  pip install evalplus\n"
        )

    lm = None
    hf_gen = None
    if args.kv_cache:
        # KV-cached HF generation path (O(n)); no torchtitan model needed.
        hf_gen = _load_hf_generate_model(
            args.hf_checkpoint, torch.float32 if args.fp32 else torch.bfloat16)
        logger.info("Protocol: evalplus-canonical chat [KV-cache HF generate]")
    else:
        model, tokenizer = _load_model(
            args.module, args.config, args.checkpoint, args.hf_checkpoint,
            early_exit_threshold=args.early_exit_threshold,
            early_exit_step=args.early_exit_step,
        )
        if args.fp32:
            # Run the whole forward in float32 to test whether bf16 precision (matmul
            # accumulation across the deep UT stack) is costing pass@1 vs the paper.
            model = model.to(torch.float32)
            logger.info("Running model in float32")
        device = next(model.parameters()).device
        # --evalplus_prompt implies chat formatting (needs _im_end_id / chat template).
        use_chat = args.chat or args.evalplus_prompt
        lm = OuroLM(model=model, tokenizer=tokenizer, device=device,
                    max_gen_toks=args.max_gen_toks, add_bos=False, chat=use_chat)
        protocol = ("evalplus-canonical chat" if args.evalplus_prompt
                    else "chat (ChatML)" if args.chat else "raw completion")
        logger.info(f"Protocol: {protocol}")
    if args.evalplus_prompt:
        sys_desc = ("none (no system turn)" if args.no_system
                    else "template-default" if args.system_prompt is None
                    else "suppressed" if args.system_prompt == ""
                    else repr(args.system_prompt))
        logger.info(f"  system_prompt={sys_desc} | prefill={not args.no_prefill}")

    problems = get_human_eval_plus()
    task_ids = sorted(problems.keys())
    if args.task_ids_file:
        wanted = [l.strip() for l in open(args.task_ids_file) if l.strip()]
        missing = [t for t in wanted if t not in problems]
        if missing:
            raise SystemExit(
                f"--task_ids_file has {len(missing)} id(s) not in HumanEval: "
                f"{missing[:5]}"
            )
        task_ids = sorted(set(wanted))
        logger.info(f"Restricted to {len(task_ids)} task_ids from {args.task_ids_file}")
    if args.limit:
        task_ids = task_ids[: args.limit]
    shard = task_ids[args.shard_index :: args.num_shards]
    logger.info(
        f"EvalPlus generate: shard {args.shard_index}/{args.num_shards} -> "
        f"{len(shard)}/{len(task_ids)} problems"
    )

    out = pathlib.Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Reset per-token UT-loop counters so avg_loops reflects only this eval
    # (adaptive-exit path; the KV-cache path runs full R=4 and doesn't track them).
    if lm is not None and hasattr(lm._model, "reset_loop_stats"):
        lm._model.reset_loop_stats()
    kv_loop_sum = 0.0    # KV-cache path tracks avg_loops here (no torchtitan model)
    kv_loop_count = 0
    t0 = time.time()
    with out.open("w") as f:
        for i, tid in enumerate(shard, 1):
            prompt = problems[tid]["prompt"]
            if args.kv_cache:
                # KV-cached HF generate (O(n)) WITH adaptive early exit; the gate
                # picks each token's output step and we accumulate avg_loops.
                solution, _ls, _lc = _generate_evalplus_chat_hf(
                    hf_gen[0], hf_gen[1], hf_gen[2], prompt, args.max_gen_toks,
                    system_prompt=args.system_prompt, prefill=not args.no_prefill,
                    no_system=args.no_system, exit_threshold=args.early_exit_threshold,
                )
                kv_loop_sum += _ls
                kv_loop_count += _lc
            elif args.evalplus_prompt:
                # Full self-contained script; submit verbatim (sanitize extracts).
                solution = _generate_evalplus_chat(
                    lm, prompt, args.max_gen_toks,
                    system_prompt=args.system_prompt, prefill=not args.no_prefill,
                    no_system=args.no_system,
                )
            elif args.chat:
                # Body that continues the prompt -> submit prompt + body.
                solution = prompt + _generate_chat_completion(
                    lm, prompt, problems[tid]["entry_point"], args.max_gen_toks
                )
            else:
                # Raw base completion -> submit prompt + completion.
                solution = prompt + _generate_completion(lm, prompt, args.max_gen_toks)
            f.write(json.dumps({"task_id": tid, "solution": solution}) + "\n")
            f.flush()  # persist per-problem so a wall-time kill keeps partial output
            if i % 10 == 0:
                logger.info(f"  {i}/{len(shard)}  ({time.time() - t0:.0f}s)")

    logger.info(f"Wrote {len(shard)} samples to {out} in {time.time() - t0:.0f}s")

    # Optional ouro_eval_summary for the boptim objective: avg_loops from the
    # model's UT-loop counters (adaptive exit) + EvalPlus base pass@1. Same JSON
    # shape the objective already parses from evaluate_humaneval.py.
    if args.eval_summary_json:
        if args.kv_cache:
            # avg_loops accumulated during KV-cache generation (per generated token)
            loop_sum = kv_loop_sum
            loop_count = kv_loop_count
            avg_loops = (kv_loop_sum / kv_loop_count) if kv_loop_count else None
        else:
            m = lm._model if lm is not None else None
            loop_sum = getattr(m, "_loop_sum", None) if m is not None else None
            loop_count = getattr(m, "_loop_count", None) if m is not None else None
            avg_loops = getattr(m, "avg_loops", None) if m is not None else None
        pass1 = ep_pass = None
        if args.evalplus_score:
            ep_pass, ep_n = _score_samples_evalplus(out)
            pass1 = (ep_pass / ep_n) if ep_n else 0.0
        summary = {
            "pass@1": pass1,
            "pass@1_evalplus_correct": ep_pass,
            "avg_loops": avg_loops,
            "loop_sum": loop_sum,
            "loop_count": loop_count,
            "n_problems": len(shard),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "early_exit_threshold": args.early_exit_threshold,
            "max_gen_toks": args.max_gen_toks,
            "protocol": "evalplus-canonical",
            "no_system": args.no_system,
        }
        with open(args.eval_summary_json, "w") as sf:
            json.dump({"ouro_eval_summary": summary}, sf, indent=2, default=str)
        logger.info(
            f"Wrote eval summary -> {args.eval_summary_json}: "
            f"pass@1={pass1} avg_loops={avg_loops}"
        )


if __name__ == "__main__":
    main()
