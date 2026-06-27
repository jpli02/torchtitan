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
        "--chat", action="store_true",
        help="Use the ChatML chat protocol (SFT-matched: apply_chat_template + "
        "body extraction) instead of raw completion. Required for the released "
        "instruct-style Ouro-1.4B, which echoes the prompt under raw completion.",
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
    device = next(model.parameters()).device
    # chat=False => raw base-completion protocol; chat=True forces add_bos and
    # uses the SFT-matched ChatML wrapper (OuroLM sets _add_bos/_im_end_id).
    lm = OuroLM(model=model, tokenizer=tokenizer, device=device,
                max_gen_toks=args.max_gen_toks, add_bos=False, chat=args.chat)
    logger.info(f"Protocol: {'chat (ChatML)' if args.chat else 'raw completion'}")

    problems = get_human_eval_plus()
    task_ids = sorted(problems.keys())
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
            if args.chat:
                completion = _generate_chat_completion(
                    lm, prompt, problems[tid]["entry_point"], args.max_gen_toks
                )
            else:
                completion = _generate_completion(lm, prompt, args.max_gen_toks)
            # EvalPlus accepts a full-program "solution"; prompt + body is the
            # standard base-completion submission (sanitize extracts the function).
            f.write(json.dumps({"task_id": tid, "solution": prompt + completion}) + "\n")
            if i % 10 == 0:
                logger.info(f"  {i}/{len(shard)}  ({time.time() - t0:.0f}s)")

    logger.info(f"Wrote {len(shard)} samples to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
