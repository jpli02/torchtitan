#!/usr/bin/env python3
"""
MBPP evaluation for TorchTitan Ouro model via lm-evaluation-harness.

Sibling of ``evaluate_humaneval.py``; it reuses that script's model loader and
``OuroLM`` wrapper (greedy decode, loop-stat accounting, sharding) and only
swaps in the MBPP-specific bits:

  * task = lm-eval ``mbpp`` (NL task description + three example asserts), scored
    by executing ``candidate + "\\n" + test`` -- so the model must return a
    *whole* function, not the indented body HumanEval wants.
  * chat protocol: wrap the rendered problem as a single user turn that asks for
    a complete fenced solution, then return the first ```python block verbatim
    (no signature-strip / body-reindent, unlike HumanEval).
  * ``--num_fewshot`` (default 3, the standard MBPP setting). The raw base path
    completes the few-shot prompt and stops at ``[DONE]``.

Usage:
    python scripts/evaluate_mbpp.py \
        --module ouro --config ouro_1_4b \
        --checkpoint ./outputs/checkpoint/step-500 \
        [--chat] [--num_fewshot 3] [--limit 20]
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from torchtitan.tools.logging import init_logger, logger

# Reuse the HumanEval evaluator's model loader + lm-eval wrapper (same scripts/
# dir, so this resolves whether run as a file or imported). Importing it also
# registers the "ouro_tt" lm-eval model once.
from evaluate_humaneval import _load_model, OuroLM


# ---------------------------------------------------------------------------
# MBPP-specific lm-eval wrapper
# ---------------------------------------------------------------------------

class MbppOuroLM(OuroLM):
    """OuroLM whose chat protocol targets MBPP (whole-function output).

    MBPP scores ``candidate + "\\n" + test`` rather than HumanEval's
    ``prompt + body``, so in chat mode we ask for a complete, self-contained
    solution and hand back the entire fenced code block. The raw (non-chat)
    path is inherited unchanged: it completes the few-shot prompt and trims at
    the task's ``[DONE]`` stop string.
    """

    @staticmethod
    def _chat_user_content(ctx: str) -> str:
        # ctx is lm-eval's rendered MBPP context: the (optional few-shot
        # examples plus) task description and the three example asserts the
        # solution must satisfy. Ask for the full program in a fenced block.
        return (
            "Write a complete, self-contained Python solution for the following "
            "task. Reply with ONLY the code inside a ```python code block; "
            "include any imports and the full function definition(s), and make "
            "sure it passes the listed assertions.\n\n" + ctx.rstrip()
        )

    def _extract_chat_body(self, text: str, req) -> str:
        # The MBPP scorer runs `candidate + "\n" + test`, so the candidate must
        # be the whole program. Return the first fenced code block verbatim (no
        # signature stripping / body re-indentation that HumanEval needs).
        m = re.search(r"```(?:python)?[ \t]*\n(.*?)(?:```|\Z)", text, re.DOTALL)
        return m.group(1) if m else text


# ---------------------------------------------------------------------------
# Eval driver (optionally sharded across GPUs), mirroring _run_humaneval
# ---------------------------------------------------------------------------

def _run_mbpp(lm, limit, num_shards, shard_index, num_fewshot):
    """Run MBPP, optionally on a stride shard of the problem set.

    num_shards <= 1 -> full set via simple_evaluate. num_shards > 1 -> slice the
    test split to problems[shard_index::num_shards] (after `limit`) so each GPU
    process scores a disjoint subset. Returns the lm-eval results dict.
    """
    import lm_eval

    if num_shards <= 1:
        return lm_eval.simple_evaluate(
            model=lm,
            tasks=["mbpp"],
            num_fewshot=num_fewshot,
            limit=limit,
            log_samples=True,
            confirm_run_unsafe_code=True,
        )

    from lm_eval import evaluator
    from lm_eval.tasks import get_task_dict, TaskManager

    task_dict = get_task_dict(["mbpp"], TaskManager())
    task = task_dict["mbpp"]
    if num_fewshot is not None:
        task.set_config(key="num_fewshot", value=int(num_fewshot))
    split = "test" if task.has_test_docs() else "validation"
    keep = list(range(len(task.dataset[split])))
    if limit:
        keep = keep[:limit]
    shard = keep[shard_index::num_shards]
    task.dataset[split] = task.dataset[split].select(shard)
    return evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=None,
        log_samples=True,
        confirm_run_unsafe_code=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MBPP eval via lm-evaluation-harness")
    parser.add_argument("--module", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="DCP checkpoint path (e.g. outputs/checkpoint/step-500)")
    parser.add_argument("--hf_checkpoint", help="HF safetensors checkpoint directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of MBPP problems")
    parser.add_argument(
        "--num_fewshot", type=int, default=3,
        help="Few-shot examples (MBPP standard is 3). Use 0 for zero-shot instruct.",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help="Split the (limited) problem set into this many disjoint shards for "
        "data-parallel eval; run one process per GPU with a distinct --shard_index.",
    )
    parser.add_argument(
        "--shard_index", type=int, default=0,
        help="Which shard (0..num_shards-1) this process evaluates: problems[shard_index::num_shards].",
    )
    parser.add_argument("--max_gen_toks", type=int, default=512)
    parser.add_argument("--output_path", default="outputs/mbpp_results.json")
    parser.add_argument(
        "--early_exit_threshold", type=float, default=None,
        help="Override model early_exit_threshold (<1.0 enables adaptive early exit).",
    )
    parser.add_argument(
        "--early_exit_step", type=int, default=None,
        help="Override model early_exit_step (force a fixed UT exit step).",
    )
    parser.add_argument(
        "--add_bos", action="store_true",
        help="Prepend a BOS (<|endoftext|>) token to each prompt. Ignored (forced on) under --chat.",
    )
    parser.add_argument(
        "--chat", action="store_true",
        help="Format prompts with the ChatML chat template (SFT-matched). Asks for a "
        "complete fenced solution and returns the code block. Off => raw few-shot "
        "completion stopping at [DONE].",
    )
    args = parser.parse_args()

    if (args.checkpoint is None) == (args.hf_checkpoint is None):
        parser.error("Provide exactly one of --checkpoint or --hf_checkpoint.")

    init_logger()

    os.environ.setdefault("ALLOW_CODE_EXECUTION", "1")
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model, tokenizer = _load_model(
        args.module,
        args.config,
        args.checkpoint,
        args.hf_checkpoint,
        early_exit_threshold=args.early_exit_threshold,
        early_exit_step=args.early_exit_step,
    )
    device = next(model.parameters()).device

    lm = MbppOuroLM(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_gen_toks=args.max_gen_toks,
        add_bos=args.add_bos,
        chat=args.chat,
    )

    # Zero the loops-per-token accumulators so avg_loops reflects only this eval.
    if hasattr(model, "reset_loop_stats"):
        model.reset_loop_stats()
    gen_t0 = time.time()
    results = _run_mbpp(lm, args.limit, args.num_shards, args.shard_index, args.num_fewshot)
    eval_time_s = time.time() - gen_t0

    # lm-eval keys metrics as "<metric>,<filter>"; MBPP's custom metric registers
    # as "pass_at_1,none" (not HumanEval's "pass@1"). Match on the metric name
    # before the comma, accepting either spelling, so the filter suffix and the
    # underscore/at variation don't break us.
    mbpp_results = results["results"]["mbpp"]
    pass_keys = [k for k in mbpp_results if k.split(",")[0] in ("pass_at_1", "pass@1")]
    if not pass_keys:
        raise KeyError(f"no pass@1 metric in mbpp results; keys={list(mbpp_results)}")
    pass_at_1 = float(mbpp_results[pass_keys[0]])

    # Average UT loops per generated token (efficiency signal). None when the
    # adaptive path never ran (e.g. threshold == 1.0 full-recurrence baseline).
    avg_loops = getattr(model, "avg_loops", None)
    total_ut_steps = getattr(model, "total_ut_steps", None)
    if avg_loops is None and total_ut_steps is not None and (
        args.early_exit_threshold is None or args.early_exit_threshold >= 1.0
    ):
        avg_loops = float(total_ut_steps)
    n_problems = len(results.get("samples", {}).get("mbpp", []))
    results["ouro_eval_summary"] = {
        "pass@1": pass_at_1,
        "eval_time_s": eval_time_s,
        "avg_loops": avg_loops,
        "total_ut_steps": total_ut_steps,
        "n_problems": n_problems,
        "loop_sum": getattr(model, "_loop_sum", None),
        "loop_count": getattr(model, "_loop_count", None),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "max_gen_toks": args.max_gen_toks,
        "early_exit_threshold": args.early_exit_threshold,
        "early_exit_step": args.early_exit_step,
        "chat": args.chat,
        "add_bos": args.add_bos or args.chat,
        "task": "mbpp",
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {out}")
    _al = "n/a" if avg_loops is None else f"{avg_loops:.3f}"
    logger.info(
        f"pass@1 = {pass_at_1:.4f}  eval_time_s = {eval_time_s:.2f}  avg_loops = {_al}"
    )


if __name__ == "__main__":
    main()
