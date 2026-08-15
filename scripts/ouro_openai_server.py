"""
scripts/ouro_openai_server.py
------------------------------
Minimal OpenAI-compatible chat-completions server for the released Ouro-1.4B,
so external harnesses (e.g. Terminal-Bench's terminus agent, via LiteLLM) can
drive it like any other OpenAI-format model.

Reuses the exact HF generation path already validated by
evaluate_humaneval_evalplus.py (_load_hf_generate_model: the RoPE/attention-
mask version shims for this custom architecture, and a KV-cached, per-token
greedy decode loop that optionally applies the adaptive early-exit gate).
Unlike that script, this builds the prompt from an ARBITRARY OpenAI-style
messages list via the tokenizer's chat template (apply_chat_template),
supporting general multi-turn conversations rather than one EvalPlus prompt.

Why not vLLM: Ouro is a Universal Transformer (weight-tied decoder layers
looped up to `total_ut_steps` times per token) with a learned per-token exit
gate choosing how many loops to run. vLLM's paged-attention engine assumes one
standard forward pass per token with uniform layer-wise KV caching across the
whole batch; it has no path for a per-token, per-request variable-depth loop,
and even vLLM's "transformers backend" fallback expects a standard forward()/
past_key_values interface Ouro's recurrence doesn't fit. This server is a
thin, single-request-at-a-time wrapper around the model's *actual* generation
logic instead -- correct, not high-throughput (fine for benchmark harnesses,
which are not throughput-sensitive).

Usage:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/ouro_openai_server.py \
        --hf_dir ./assets/hf/Ouro-1.4B --port 8009
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_humaneval_evalplus import _load_hf_generate_model, _ouro_exit_steps  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402
import uvicorn  # noqa: E402


# ===========================================================================
# Generation (mirrors _generate_evalplus_chat_hf's decode loop, but built from
# an arbitrary chat-template prompt rather than the EvalPlus-specific one).
# ===========================================================================
@torch.no_grad()
def generate_chat(hf_model, hf_tok, eos_ids, messages, max_new, exit_threshold=None):
    prompt_str = hf_tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    dev = hf_model.device
    enc = hf_tok(prompt_str, return_tensors="pt", add_special_tokens=False)
    cur = enc.input_ids.to(dev)
    n_prompt = cur.shape[1]
    L = n_prompt
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
        if thr is not None:
            es_last = int(_ouro_exit_steps(gate_list, thr)[0, -1].item())
        else:
            es_last = R - 1  # full recurrence (pretrained base-model setting)
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
    avg_loops = (loop_sum / loop_count) if loop_count else 0.0
    return text, n_prompt, len(new_ids), avg_loops


# ===========================================================================
# OpenAI-compatible schema (subset: only what a terminus-style agent needs)
# ===========================================================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = 1024
    temperature: float | None = 0.0
    stream: bool | None = False


app = FastAPI()
_state: dict[str, Any] = {}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(400, "streaming is not supported by this server")
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    max_new = int(req.max_tokens or 1024)

    loop = asyncio.get_running_loop()
    async with _state["lock"]:
        text, n_prompt, n_completion, avg_loops = await loop.run_in_executor(
            None,
            functools.partial(
                generate_chat,
                _state["model"], _state["tok"], _state["eos_ids"],
                messages, max_new, _state["exit_threshold"],
            ),
        )
        # A multi-turn agent harness sends a growing conversation each call, so
        # peak KV-cache size grows across a session. PyTorch's caching allocator
        # doesn't release blocks back to the driver on its own once reserved for
        # a large sequence, so nvidia-smi's "used" number ratchets up over a long
        # unattended run (observed 6GB -> 43GB over 12 requests). Free it back
        # every request -- cheap relative to a ~30s generation call, and keeps
        # memory bounded for an 80-task run instead of trending toward OOM.
        await loop.run_in_executor(None, torch.cuda.empty_cache)

    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": now,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": n_prompt,
            "completion_tokens": n_completion,
            "total_tokens": n_prompt + n_completion,
        },
        # non-standard extra field; harmless for OpenAI-schema consumers that
        # ignore unknown keys, useful for our own inspection of avg_loops.
        "ouro_avg_loops": avg_loops,
    }


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": _state["model_name"], "object": "model"}]}


@app.get("/health")
async def health():
    return {"status": "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dir", default="./assets/hf/Ouro-1.4B")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model_name", default="ouro-1.4b")
    ap.add_argument(
        "--early_exit_threshold", type=float, default=None,
        help="None/>=1.0 = full R=4 recurrence (pretrained base-model setting, "
        "the default). <1.0 enables the adaptive early-exit gate.",
    )
    ap.add_argument("--attn_impl", default="eager")
    args = ap.parse_args()

    model, tok, eos_ids = _load_hf_generate_model(
        args.hf_dir, dtype=torch.float32, attn_impl=args.attn_impl)
    _state["model"] = model
    _state["tok"] = tok
    _state["eos_ids"] = eos_ids
    _state["model_name"] = args.model_name
    _state["exit_threshold"] = args.early_exit_threshold
    _state["lock"] = asyncio.Lock()

    print(f"[ouro-server] ready: model={args.model_name} "
          f"exit_threshold={args.early_exit_threshold} on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
