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


def _load_model(module, config_name, checkpoint_path, hf_checkpoint_path):
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
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length
        self._max_gen_toks = max_gen_toks
        self._batch_size = batch_size

    # --- required properties ------------------------------------------------

    @property
    def eot_token_id(self) -> int:
        return getattr(self._tokenizer, "eos_id", 2)

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
    ) -> torch.Tensor:
        generated = input_ids
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self._model(generated)
                next_tok = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                generated = torch.cat(
                    [generated, torch.tensor([[next_tok]], device=self._device)], dim=1
                )
                if next_tok == self.eot_token_id:
                    break
                # check stop strings on newly generated text
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

            # Encode stop strings to token IDs for early exit detection
            stop_ids = [
                self._tokenizer.encode(s, add_bos=False, add_eos=False)
                for s in until
            ]

            input_ids = torch.tensor(
                self._tokenizer.encode(ctx, add_bos=True, add_eos=False),
                dtype=torch.long,
                device=self._device,
            ).unsqueeze(0)

            # Truncate prompt if too long
            if input_ids.shape[1] > self._max_length - max_new:
                input_ids = input_ids[:, -(self._max_length - max_new):]

            if (i + 1) % 10 == 0:
                logger.info(f"HumanEval generate {i + 1}/{len(requests)}")
                sys.stdout.flush()

            gen_ids = self._greedy_until(input_ids, stop_ids, max_new)
            text = self._tokenizer.decode(gen_ids[0].tolist())

            # Trim at any stop string
            for s in until:
                if s in text:
                    text = text[: text.index(s)]

            results.append(text)
        return results

    # --- log-likelihood (required by lm-eval) --------------------------------

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        results = []
        for req in requests:
            ctx, cont = req.args
            full = ctx + cont
            input_ids = torch.tensor(
                self._tokenizer.encode(full, add_bos=True, add_eos=False),
                dtype=torch.long,
                device=self._device,
            ).unsqueeze(0)
            ctx_len = len(self._tokenizer.encode(ctx, add_bos=True, add_eos=False))

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
                self._tokenizer.encode(text, add_bos=True, add_eos=True),
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HumanEval eval via lm-evaluation-harness")
    parser.add_argument("--module", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="DCP checkpoint path (e.g. outputs/checkpoint/step-500)")
    parser.add_argument("--hf_checkpoint", help="HF safetensors checkpoint directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of HumanEval problems")
    parser.add_argument("--max_gen_toks", type=int, default=512)
    parser.add_argument("--output_path", default="outputs/humaneval_results.json")
    args = parser.parse_args()

    if (args.checkpoint is None) == (args.hf_checkpoint is None):
        parser.error("Provide exactly one of --checkpoint or --hf_checkpoint.")

    init_logger()

    import os
    os.environ.setdefault("ALLOW_CODE_EXECUTION", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model, tokenizer = _load_model(
        args.module, args.config, args.checkpoint, args.hf_checkpoint
    )
    device = next(model.parameters()).device

    lm = OuroLM(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_gen_toks=args.max_gen_toks,
    )

    import lm_eval
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["humaneval"],
        limit=args.limit,
        log_samples=True,
    )

    import json, pathlib
    out = pathlib.Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {out}")
    logger.info(f"pass@1 = {results['results']['humaneval']['pass@1']:.4f}")


if __name__ == "__main__":
    main()
