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
    user = f"{_EVALPLUS_INSTRUCTION}\n```\n{prompt.strip()}\n```\n"
    if no_system:
        # Template always forces a system turn; emit ChatML by hand without one.
        prompt_str = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    else:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user})
        prompt_str = lm._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
    if prefill:
        prompt_str = prompt_str + f"{_EVALPLUS_RESPONSE}\n```python\n"
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
    p.add_argument("--output_path", default="outputs/he_evalplus/samples.jsonl")
    args = p.parse_args()

    if (args.checkpoint is None) == (args.hf_checkpoint is None):
        p.error("Provide exactly one of --checkpoint or --hf_checkpoint.")

    init_logger()

    try:
        from evalplus.data import get_human_eval_plus
    except ImportError:
        raise SystemExit(
            "EvalPlus is not installed.\n  Run:  pip install evalplus\n"
        )

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
    t0 = time.time()
    with out.open("w") as f:
        for i, tid in enumerate(shard, 1):
            prompt = problems[tid]["prompt"]
            if args.evalplus_prompt:
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


if __name__ == "__main__":
    main()
