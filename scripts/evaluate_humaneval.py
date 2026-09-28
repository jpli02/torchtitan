#!/usr/bin/env python3
"""
HumanEval evaluation for TorchTitan Ouro model via lm-evaluation-harness.

Usage:
    python scripts/evaluate_humaneval.py \
        --module ouro --config ouro_1_4b \
        --checkpoint ./outputs/checkpoint/step-500 \
        [--limit 20] [--batch_size 1]
"""

import argparse
import copy
import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import device_module, device_type


# ---------------------------------------------------------------------------
# Model loading (mirrors evaluate_gsm8k.py)
# ---------------------------------------------------------------------------

def _apply_hf_config_json_overrides(model_config, hf_dir: Path):
    cfg_path = hf_dir / "config.json"
    if not cfg_path.is_file():
        return model_config
    with cfg_path.open() as f:
        hf = json.load(f)
    overrides: dict = {}
    for key in ("early_exit_threshold", "total_ut_steps"):
        if key in hf:
            overrides[key] = type(getattr(model_config, key))(hf[key])
    if overrides:
        logger.info(f"HF config.json overrides: {overrides}")
    return dataclasses.replace(model_config, **overrides) if overrides else model_config


def _load_model(
    module,
    config_name,
    checkpoint_path,
    hf_checkpoint_path,
    early_exit_threshold=None,
    early_exit_step=None,
):
    config_manager = ConfigManager()
    config = config_manager.parse_args(["--module", module, "--config", config_name])

    device = torch.device(f"{device_type}:0")
    device_module.set_device(device)

    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    tokenizer = HuggingFaceTokenizer.Config().build(
        tokenizer_path=config.hf_assets_path
    )

    model_config = copy.deepcopy(config.model_spec.model)
    model_config.update_from_config(trainer_config=config)
    if hf_checkpoint_path is not None:
        model_config = _apply_hf_config_json_overrides(
            model_config, Path(hf_checkpoint_path).resolve()
        )
    # Adaptive early-exit overrides: a threshold < 1.0 makes the trained gate
    # govern how many UT steps run at inference (see OuroModel._adaptive_forward),
    # so generation latency reflects the gate's learned exit behavior.
    exit_overrides: dict = {}
    if early_exit_threshold is not None:
        exit_overrides["early_exit_threshold"] = float(early_exit_threshold)
    if early_exit_step is not None:
        exit_overrides["early_exit_step"] = int(early_exit_step)
    if exit_overrides:
        logger.info(f"Early-exit overrides: {exit_overrides}")
        model_config = dataclasses.replace(model_config, **exit_overrides)

    with torch.device(device):
        model = model_config.build()
    model.to(device)
    model.eval()
    with torch.no_grad():
        model.init_weights()

    if checkpoint_path is not None:
        state_dict = model.state_dict()
        logger.info(f"Loading DCP checkpoint from {checkpoint_path}")
        t0 = time.time()
        dcp.load(state_dict, checkpoint_id=checkpoint_path)
        logger.info(f"DCP checkpoint loaded in {time.time() - t0:.2f}s")
    elif hf_checkpoint_path is not None:
        sd_adapter = config.model_spec.state_dict_adapter(model_config, hf_checkpoint_path)
        if sd_adapter is None:
            raise RuntimeError("Model does not provide a state_dict_adapter for HF loading.")
        state_dict = model.state_dict()
        hf_state_dict = sd_adapter.to_hf(state_dict)
        logger.info(f"Loading HF checkpoint from {hf_checkpoint_path}")
        t0 = time.time()
        dcp.load(hf_state_dict, storage_reader=HuggingFaceStorageReader(path=hf_checkpoint_path))
        tt_state_dict = sd_adapter.from_hf(hf_state_dict)
        model.load_state_dict(tt_state_dict, strict=True)
        logger.info(f"HF checkpoint loaded in {time.time() - t0:.2f}s")
    else:
        raise ValueError("Provide exactly one of --checkpoint or --hf_checkpoint.")

    return model, tokenizer


def _evalplus_base_score(samples, workdir):
    """Re-score lm-eval HumanEval generations with EvalPlus for a cleaner pass@1.

    evalplus.sanitize (AST-based extraction of the entry function) then exec each
    sanitized solution against the problem's *base* HumanEval test with a
    per-problem timeout. Subset-safe: does NOT call evalplus.evaluate (which
    asserts all 164 problems are present), so it works with --limit / shards.
    Returns (n_pass, n_total) where n_total is the number of generated samples
    (a problem sanitize drops entirely counts as a fail). Generation and the
    avg_loops accounting are untouched -- only scoring changes.
    """
    import json as _json, subprocess, sys, signal, pathlib
    from evalplus.data import get_human_eval_plus
    prob = get_human_eval_plus()
    workdir = pathlib.Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    raw = workdir / "ep_samples.jsonl"
    with raw.open("w") as f:
        for s in samples:
            doc = s["doc"]; tid = doc["task_id"]
            sol = (s.get("filtered_resps") or [""])[0]
            if isinstance(sol, list):
                sol = sol[0] if sol else ""
            # lm-eval's filtered resp is usually the full program; if it lacks the
            # entry function, prepend the prompt so sanitize sees a definition.
            if f"def {doc['entry_point']}" not in sol:
                sol = doc["prompt"] + sol
            f.write(_json.dumps({"task_id": tid, "solution": sol}) + "\n")
    subprocess.run([sys.executable, "-m", "evalplus.sanitize", "--samples", str(raw)],
                   check=True, cwd=str(workdir))
    san = raw.with_name(raw.stem + "-sanitized.jsonl")
    by_tid = {_json.loads(l)["task_id"]: _json.loads(l)["solution"]
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
    return npass, len(samples)


# ---------------------------------------------------------------------------
# lm-eval model wrapper
# ---------------------------------------------------------------------------

try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ImportError:
    raise SystemExit(
        "lm-evaluation-harness is not installed.\n"
        "Run:  pip install 'lm_eval[code]'\n"
    )


@register_model("ouro_tt")
class OuroLM(LM):
    """Thin lm-eval wrapper around a TorchTitan Ouro model."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
        max_length: int = 2048,
        max_gen_toks: int = 512,
        batch_size: int = 1,
        add_bos: bool = False,
        chat: bool = False,
        no_system: bool = False,
        chat_add_bos: bool = False,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length
        self._max_gen_toks = max_gen_toks
        self._batch_size = batch_size
        # Ouro's tokenizer (StarCoder2/SmolLM family) sets add_bos_token=False;
        # bos==eos==unk==<|endoftext|> (id 0). The HF quick-start tokenizes with
        # no leading BOS, so prepending one inserts a document-boundary token
        # the base model never sees before a completion and degrades greedy
        # decoding. Default off; flip via --add_bos to A/B test.
        #
        # Chat mode (--chat) wraps prompts in ChatML via apply_chat_template. The
        # released base model's chat template starts directly with <|im_start|>
        # and its tokenizer is add_bos_token=False, so the *native* chat usage
        # adds no leading BOS -- that is now the default (chat_add_bos=False).
        # --chat_force_bos restores the legacy leading-BOS behavior for A/B.
        self._chat = chat
        self._no_system = no_system
        self._add_bos = chat_add_bos if chat else add_bos
        # `<|im_end|>` (id 2) closes an assistant turn in ChatML; stop on it as
        # well as EOS when generating chat completions.
        self._im_end_id = self._tokenizer.token_to_id("<|im_end|>") if chat else None

    # --- required properties ------------------------------------------------

    @property
    def eot_token_id(self) -> int:
        # getattr(.., "eos_id", default) returns the default only when the
        # attribute is absent; this tokenizer *has* eos_id but it is None, so
        # fall back through bos_id (also <|endoftext|>) to the literal id 0.
        for attr in ("eos_id", "bos_id"):
            val = getattr(self._tokenizer, attr, None)
            if val is not None:
                return val
        return 0

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return self._device

    # --- generation ---------------------------------------------------------

    def _greedy_until(
        self,
        input_ids: torch.Tensor,
        stop_ids: list[list[int]],
        max_new_tokens: int,
        extra_stop_token_ids: set[int] | None = None,
    ) -> torch.Tensor:
        generated = input_ids
        prompt_len = input_ids.shape[1]
        stop_token_ids = {self.eot_token_id}
        if extra_stop_token_ids:
            stop_token_ids |= extra_stop_token_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self._model(generated)
                next_tok = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                generated = torch.cat(
                    [generated, torch.tensor([[next_tok]], device=self._device)], dim=1
                )
                if next_tok in stop_token_ids:
                    break
                # check stop strings on newly generated text
                if stop_ids:
                    gen_text = self._tokenizer.decode(
                        generated[0, prompt_len:].tolist()
                    )
                    if any(
                        self._tokenizer.decode(s) in gen_text
                        for s in stop_ids
                        if s
                    ):
                        break
        return generated[:, prompt_len:]

    def generate_until(self, requests) -> list[str]:
        results = []
        for i, req in enumerate(requests):
            ctx, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_new = min(
                gen_kwargs.get("max_gen_toks", self._max_gen_toks),
                self._max_gen_toks,
            )

            if self._chat:
                # ChatML completion (SFT-matched). The base `until` stop strings
                # (\nclass, \ndef, ...) would prematurely cut a multi-line fenced
                # solution, so we don't use them here; the assistant turn ends at
                # <|im_end|>/EOS instead.
                if self._no_system:
                    # The ChatML template always injects a default system turn
                    # ("You are a helpful assistant."); build the prompt by hand
                    # without one (higher HumanEval for this base model).
                    prompt_str = (
                        f"<|im_start|>user\n{self._chat_user_content(ctx)}"
                        f"<|im_end|>\n<|im_start|>assistant\n"
                    )
                else:
                    prompt_str = self._tokenizer.apply_chat_template(
                        [{"role": "user", "content": self._chat_user_content(ctx)}],
                        add_generation_prompt=True,
                    )
                stop_ids: list[list[int]] = []
                extra_stop = {self._im_end_id} if self._im_end_id is not None else set()
            else:
                prompt_str = ctx
                # Encode stop strings to token IDs for early exit detection
                stop_ids = [
                    self._tokenizer.encode(s, add_bos=False, add_eos=False)
                    for s in until
                ]
                extra_stop = set()

            input_ids = torch.tensor(
                self._tokenizer.encode(prompt_str, add_bos=self._add_bos, add_eos=False),
                dtype=torch.long,
                device=self._device,
            ).unsqueeze(0)

            # Truncate prompt if too long
            if input_ids.shape[1] > self._max_length - max_new:
                input_ids = input_ids[:, -(self._max_length - max_new):]

            if (i + 1) % 10 == 0:
                logger.info(f"HumanEval generate {i + 1}/{len(requests)}")
                sys.stdout.flush()

            gen_ids = self._greedy_until(
                input_ids, stop_ids, max_new, extra_stop_token_ids=extra_stop
            )
            text = self._tokenizer.decode(gen_ids[0].tolist())

            if self._chat:
                # The HumanEval scorer composes `doc["prompt"] + completion`
                # (lm_eval humaneval build_predictions), so reduce the chat
                # response to the function body that continues the prompt.
                text = self._extract_chat_body(text, req)
            else:
                # Trim at any stop string
                for s in until:
                    if s in text:
                        text = text[: text.index(s)]

            results.append(text)
        return results

    @staticmethod
    def _chat_user_content(ctx: str) -> str:
        """Wrap a HumanEval prompt as a single user instruction (body-only).

        The SFT data (OpenHands trajectories) has no function-completion turn,
        so there is no exact template to copy; we mirror the *format* (a ChatML
        user turn the model answers). We ask for **only the function body** (not
        a regenerated signature/docstring) so the model spends its token budget
        and the no-KV-cache greedy loop's wall-clock on the implementation, not
        on re-emitting the prompt — important across a BO sweep. The extractor
        still tolerates a disobedient full-function response.
        """
        return (
            "Complete the following Python function. Reply with ONLY the "
            "function body (the indented statements that go after the signature "
            "and docstring) inside a ```python code block; do not repeat the "
            "signature or docstring.\n\n```python\n" + ctx.rstrip() + "\n```"
        )

    def _extract_chat_body(self, text: str, req) -> str:
        """Reduce a chat response to the indented body that follows the prompt.

        ``build_predictions`` prepends ``doc["prompt"]`` (signature + docstring),
        so the returned completion must be the *indented body*. Steps:
          1. Take the first fenced code block (or the raw text).
          2. If the model disobeyed and re-emitted ``def <entry_point>(...):``,
             drop through that signature line (a re-emitted docstring after it is
             a harmless bare string expression).
          3. Ensure the body is indented under the function: if its first
             non-empty line is indented < 4 spaces, indent every non-blank line
             by 4 spaces (preserving relative structure) so ``prompt + body`` is
             valid Python.
        """
        entry_point = ""
        try:
            entry_point = req.doc.get("entry_point", "")  # lm-eval attaches doc
        except Exception:
            pass

        m = re.search(r"```(?:python)?[ \t]*\n(.*?)(?:```|\Z)", text, re.DOTALL)
        code = m.group(1) if m else text

        if entry_point:
            lines = code.splitlines()
            for idx, line in enumerate(lines):
                if re.match(rf"\s*def\s+{re.escape(entry_point)}\b", line):
                    # Skip the (possibly multi-line) signature up to its ':'.
                    sig_end = idx
                    while sig_end < len(lines) and not lines[sig_end].rstrip().endswith(":"):
                        sig_end += 1
                    code = "\n".join(lines[sig_end + 1:])
                    break

        return self._ensure_body_indent(code)

    @staticmethod
    def _ensure_body_indent(code: str, indent: str = "    ") -> str:
        """Indent a function body by 4 spaces if it came back at column 0.

        Body-only responses often omit the leading indentation; without it,
        ``prompt + body`` would put statements outside the function. If the
        first non-empty line is under-indented, shift all non-blank lines right
        by one level, preserving relative indentation.
        """
        lines = code.split("\n")
        first = next((ln for ln in lines if ln.strip()), None)
        if first is None:
            return code
        leading = len(first) - len(first.lstrip())
        if leading >= len(indent):
            return code
        return "\n".join((indent + ln) if ln.strip() else ln for ln in lines)

    # --- log-likelihood (required by lm-eval) --------------------------------

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        results = []
        for req in requests:
            ctx, cont = req.args
            full = ctx + cont
            input_ids = torch.tensor(
                self._tokenizer.encode(full, add_bos=self._add_bos, add_eos=False),
                dtype=torch.long,
                device=self._device,
            ).unsqueeze(0)
            ctx_len = len(self._tokenizer.encode(ctx, add_bos=self._add_bos, add_eos=False))

            with torch.no_grad():
                logits = self._model(input_ids)

            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            # Continuation tokens: positions ctx_len-1 .. end-1 predict ctx_len .. end
            cont_ids = input_ids[0, ctx_len:]
            lp = log_probs[0, ctx_len - 1 : ctx_len - 1 + len(cont_ids)]
            gathered = lp.gather(1, cont_ids.unsqueeze(1)).squeeze(1)
            ll = float(gathered.sum())
            is_greedy = bool(
                (log_probs[0, ctx_len - 1 :].argmax(-1)[:len(cont_ids)] == cont_ids).all()
            )
            results.append((ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests) -> list[float]:
        results = []
        for req in requests:
            (text,) = req.args
            input_ids = torch.tensor(
                self._tokenizer.encode(text, add_bos=self._add_bos, add_eos=True),
                dtype=torch.long,
                device=self._device,
            ).unsqueeze(0)
            with torch.no_grad():
                logits = self._model(input_ids)
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            targets = input_ids[0, 1:]
            lp = log_probs[0, :-1].gather(1, targets.unsqueeze(1)).squeeze(1)
            results.append(float(lp.sum()))
        return results


# ---------------------------------------------------------------------------
# Evaluation driver (supports data-parallel sharding)
# ---------------------------------------------------------------------------

def _run_humaneval(lm, limit, num_shards, shard_index):
    """Run HumanEval, optionally on a stride shard of the problem set.

    num_shards <= 1  -> full set via simple_evaluate (unchanged path).
    num_shards  > 1  -> slice the task's test split to problems[shard_index::
    num_shards] (after applying `limit`) and run the lower-level evaluator so
    each GPU process scores only its disjoint subset. Returns the lm-eval
    results dict.
    """
    import lm_eval

    if num_shards <= 1:
        return lm_eval.simple_evaluate(
            model=lm,
            tasks=["humaneval"],
            limit=limit,
            log_samples=True,
            confirm_run_unsafe_code=True,
        )

    from lm_eval import evaluator
    from lm_eval.tasks import get_task_dict, TaskManager

    task_dict = get_task_dict(["humaneval"], TaskManager())
    task = task_dict["humaneval"]
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
    parser = argparse.ArgumentParser(description="HumanEval eval via lm-evaluation-harness")
    parser.add_argument("--module", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="DCP checkpoint path (e.g. outputs/checkpoint/step-500)")
    parser.add_argument("--hf_checkpoint", help="HF safetensors checkpoint directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of HumanEval problems")
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
    parser.add_argument(
        "--evalplus_score", action="store_true",
        help="Re-score the generated solutions with EvalPlus (sanitize + base "
        "HumanEval test exec) and report that as pass@1 instead of lm-eval's raw "
        "score; generation and avg_loops are unchanged. Subset-safe (limit/shards).",
    )
    parser.add_argument("--output_path", default="outputs/humaneval_results.json")
    parser.add_argument(
        "--early_exit_threshold",
        type=float,
        default=None,
        help="Override model early_exit_threshold (<1.0 enables adaptive early exit).",
    )
    parser.add_argument(
        "--early_exit_step",
        type=int,
        default=None,
        help="Override model early_exit_step (force a fixed UT exit step).",
    )
    parser.add_argument(
        "--add_bos",
        action="store_true",
        help="Prepend a BOS (<|endoftext|>) token to each prompt. Off by "
        "default: Ouro's tokenizer uses add_bos_token=False, and a leading "
        "BOS degrades greedy completion. Ignored (forced on) under --chat.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Format prompts with the ChatML chat template (SFT-matched: "
        "apply_chat_template + add_bos). Use for the instruction/SFT'd model; "
        "stops at <|im_end|>/EOS and extracts the function body from the "
        "assistant turn. Off => raw base-completion protocol.",
    )
    parser.add_argument(
        "--chat_force_bos",
        action="store_true",
        help="Under --chat, prepend a leading BOS before the ChatML template "
        "(legacy behavior). Default: no BOS, matching the released model's "
        "add_bos_token=False native chat usage.",
    )
    parser.add_argument(
        "--no_system",
        action="store_true",
        help="Under --chat, remove the system turn entirely (the ChatML template "
        "otherwise injects a default 'You are a helpful assistant.'). Raises "
        "HumanEval for this base model.",
    )
    args = parser.parse_args()

    if (args.checkpoint is None) == (args.hf_checkpoint is None):
        parser.error("Provide exactly one of --checkpoint or --hf_checkpoint.")

    init_logger()

    import os
    os.environ.setdefault("ALLOW_CODE_EXECUTION", "1")
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

    lm = OuroLM(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_gen_toks=args.max_gen_toks,
        add_bos=args.add_bos,
        chat=args.chat,
        no_system=args.no_system,
        chat_add_bos=args.chat_force_bos,
    )

    import lm_eval
    # Zero the loops-per-token accumulators so avg_loops reflects only this eval.
    if hasattr(model, "reset_loop_stats"):
        model.reset_loop_stats()
    # Wall-clock of generation+scoring only (excludes model load), used as the
    # efficiency signal by the Bayesian-optimization objective.
    gen_t0 = time.time()
    # humaneval executes model-generated code to score pass@1; lm-eval gates this
    # behind an explicit opt-in (HF_ALLOW_CODE_EVAL=1 covers the metric). Trusted
    # checkpoints on a trusted cluster. num_shards>1 evaluates problems[
    # shard_index::num_shards] for data-parallel eval across GPUs.
    results = _run_humaneval(lm, args.limit, args.num_shards, args.shard_index)
    eval_time_s = time.time() - gen_t0

    # lm-eval reports metrics keyed by "<metric>,<filter>" (e.g. "pass@1,create_test"),
    # not a bare "pass@1", and the filter suffix varies across versions. Match on the
    # metric name before the comma so we don't break on the suffix.
    he_results = results["results"]["humaneval"]
    try:
        pass_at_1 = float(he_results["pass@1"])
    except KeyError:
        pass_keys = [k for k in he_results if k.split(",")[0] == "pass@1"]
        if not pass_keys:
            raise KeyError(
                f"no pass@1 metric in humaneval results; keys={list(he_results)}"
            )
        pass_at_1 = float(he_results[pass_keys[0]])
    # Compact, stable summary the BO objective parses (full lm-eval dump is large
    # and schema-variable across versions).
    # Average UT loops per generated token (efficiency signal). None when the
    # adaptive path never ran (e.g. threshold == 1.0 full-recurrence baseline,
    # where every token uses total_ut_steps loops by construction).
    avg_loops = getattr(model, "avg_loops", None)
    total_ut_steps = getattr(model, "total_ut_steps", None)
    if avg_loops is None and total_ut_steps is not None and (
        args.early_exit_threshold is None or args.early_exit_threshold >= 1.0
    ):
        avg_loops = float(total_ut_steps)
    # Raw loop accumulators + this shard's problem count so a parent process can
    # aggregate across shards exactly: pass@1 = Sum(correct)/Sum(problems),
    # avg_loops = Sum(loop_sum)/Sum(loop_count).
    n_problems = len(results.get("samples", {}).get("humaneval", []))
    # Optional EvalPlus re-scoring: replace lm-eval's raw pass@1 with the
    # sanitize+base-test number. avg_loops / loop accumulators are untouched, so
    # the efficiency signal stays identical to a non-EvalPlus run.
    ep_fields: dict = {}
    if getattr(args, "evalplus_score", False):
        import pathlib as _pl
        _samples = results.get("samples", {}).get("humaneval", [])
        _ep_pass, _ep_n = _evalplus_base_score(
            _samples, _pl.Path(args.output_path).parent / "evalplus_rescore")
        ep_fields = {"pass@1_lmeval": pass_at_1, "pass@1_evalplus_correct": _ep_pass}
        pass_at_1 = (_ep_pass / _ep_n) if _ep_n else 0.0
        logger.info(
            f"EvalPlus re-score: base pass@1 = {pass_at_1:.4f} ({_ep_pass}/{_ep_n})"
            f"  [lm-eval was {ep_fields['pass@1_lmeval']:.4f}]"
        )
    results["ouro_eval_summary"] = {
        "pass@1": pass_at_1,
        **ep_fields,
        "eval_time_s": eval_time_s,
        "avg_loops": avg_loops,
        "total_ut_steps": total_ut_steps,
        "n_problems": n_problems,
        "loop_sum": getattr(model, "_loop_sum", None),
        "loop_count": getattr(model, "_loop_count", None),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "limit": args.limit,
        "max_gen_toks": args.max_gen_toks,
        "early_exit_threshold": args.early_exit_threshold,
        "early_exit_step": args.early_exit_step,
        "chat": args.chat,
        "add_bos": args.add_bos or args.chat,
    }

    import json, pathlib
    out = pathlib.Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {out}")
    _al = "n/a" if avg_loops is None else f"{avg_loops:.3f}"
    logger.info(
        f"pass@1 = {pass_at_1:.4f}  eval_time_s = {eval_time_s:.2f}  "
        f"avg_loops = {_al}"
    )


if __name__ == "__main__":
    main()
