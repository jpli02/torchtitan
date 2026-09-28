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
def _repeated_ngram_block(new_ids: list[int], n: int) -> set[int]:
    """Token ids that would complete an n-gram already seen in new_ids.

    Standard no-repeat-ngram constraint (as in HF's NoRepeatNGramLogitsProcessor):
    if the last (n-1) generated tokens have appeared as an n-gram prefix before,
    whatever token followed them previously is banned this step. Greedy argmax
    decoding with no such guard is prone to repetition loops on structured/
    repetitive outputs (e.g. a JSON array of near-identical command objects) --
    observed here as the agent-harness model looping the same command forever
    and never closing its JSON before the token budget runs out. HumanEval's
    short-completion protocol (evaluate_humaneval_evalplus.py) doesn't hit this
    and is left untouched; this only guards the general chat-completion path.
    """
    if n <= 0 or len(new_ids) < n:
        return set()
    prefix = tuple(new_ids[-(n - 1):])
    blocked = set()
    for i in range(len(new_ids) - n + 1):
        if tuple(new_ids[i:i + n - 1]) == prefix:
            blocked.add(new_ids[i + n - 1])
    return blocked


@torch.no_grad()
def generate_chat(hf_model, hf_tok, eos_ids, messages, max_new, exit_threshold=None,
                   no_repeat_ngram_size=3, session=None, temperature=0.0, top_p=1.0):
    """session (optional, mutated in place): {"ids": [...], "cache": DynamicCache}
    from the PREVIOUS call. A multi-turn agent harness resends the whole growing
    conversation every turn; without reuse, each turn is a full fresh prefill of
    an ever-longer prompt, making cumulative cost across a session O(turns^2) --
    e.g. one real terminal-bench task hit 33 turns with prompt_tokens growing
    1125 -> 33285 and per-turn latency growing from ~90s to ~192s, timing out a
    3600s task budget even after a 5.6x raw-decode speedup (fp32/eager ->
    bf16/sdpa) made no dent, because the bottleneck was repeated O(n) prefill,
    not per-token decode speed. If the new prompt is an exact token-for-token
    continuation of the cached one, reuse its KV cache and only prefill the new
    suffix; otherwise (new conversation, or the harness's history diverged)
    fall back to a full fresh prefill -- always correct, just not always fast.
    """
    # enable_thinking=False: the template only inserts a literal '<think>\n'
    # opener inside the add_generation_prompt block, i.e. only for a FRESH
    # generation slot -- a completed assistant turn replayed later as history
    # never gets one back, regardless of this flag (structural template
    # asymmetry, not a flag default). That made every KV-cache prefix check
    # fail ("MISMATCH at token 32/82: cached=...<think>... new=..." -- the
    # model still reasons in plain text either way, so this only removes a
    # decorative tag, not the reasoning itself) once turn 2's history
    # reconstruction was compared against turn 1's cached generation prompt.
    # Disabling it makes both tokenizations agree so the cache can actually
    # be reused across turns.
    prompt_str = hf_tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    print(f"[ouro-server] DEBUG prompt tail: {prompt_str[-80:]!r}", flush=True)
    dev = hf_model.device
    enc = hf_tok(prompt_str, return_tensors="pt", add_special_tokens=False)
    full_ids = enc.input_ids[0].tolist()
    n_prompt = len(full_ids)

    prev_ids = session.get("ids") if session else None
    if prev_ids and len(prev_ids) < n_prompt and full_ids[:len(prev_ids)] == prev_ids:
        cache = session["cache"]
        start = len(prev_ids)
        cur = torch.tensor([full_ids[start:]], device=dev)
        L = n_prompt
        cache_position = torch.arange(start, L, device=dev)
        print(f"[ouro-server] KV-cache HIT: reusing {start} cached tokens, "
              f"prefilling {n_prompt - start} new", flush=True)
    else:
        cache = None
        cur = enc.input_ids.to(dev)
        L = n_prompt
        cache_position = torch.arange(L, device=dev)
        miss_reason = ("no session" if not prev_ids else
                        "prompt shorter than cache" if len(prev_ids) >= n_prompt else
                        "prefix mismatch")
        if miss_reason == "prefix mismatch":
            div = next(i for i in range(min(len(prev_ids), n_prompt))
                       if prev_ids[i] != full_ids[i])
            print(f"[ouro-server] MISMATCH at token {div}/{len(prev_ids)}: "
                  f"cached={prev_ids[max(0,div-3):div+3]!r} "
                  f"new={full_ids[max(0,div-3):div+3]!r} "
                  f"cached_txt={hf_tok.decode(prev_ids[max(0,div-8):div+8])!r} "
                  f"new_txt={hf_tok.decode(full_ids[max(0,div-8):div+8])!r}",
                  flush=True)
        print(f"[ouro-server] KV-cache MISS ({miss_reason}): full prefill of "
              f"{n_prompt} tokens", flush=True)
    new_ids = []
    loop_sum = 0.0
    loop_count = 0
    hit_eos = False
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
        logits = hf_model.lm_head(hidden_list[es_last][:, -1:, :])[0, -1]
        blocked = _repeated_ngram_block(new_ids, no_repeat_ngram_size)
        if temperature and temperature > 0.0:
            # Temperature/top-p sampling. Needed for any pass@k evaluation: with
            # greedy argmax the model is a deterministic function of its context,
            # so k attempts at the same task return byte-identical answers and
            # pass@k collapses to pass@1. Agent harnesses also *ask* for this --
            # terminal-bench's terminus-2 sends temperature=0.7 on every request
            # -- so honouring it is closer to the contract they expect than
            # silently decoding greedily.
            filt = logits.float() / temperature
            if blocked:
                filt[list(blocked)] = float("-inf")
            probs = torch.softmax(filt, dim=-1)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                # Keep the smallest prefix whose mass exceeds top_p; the shift
                # keeps the first token even when it alone already exceeds it.
                cutoff = cum - sorted_probs > top_p
                sorted_probs[cutoff] = 0.0
                total = sorted_probs.sum()
                if total <= 0:  # numerical degenerate case: fall back to argmax
                    nt = int(torch.argmax(probs))
                    loop_sum += es_last + 1
                    loop_count += 1
                    if nt in eos_ids:
                        hit_eos = True
                        break
                    new_ids.append(nt)
                    cur = torch.tensor([[nt]], device=dev)
                    cache_position = torch.tensor([L + step], device=dev)
                    continue
                sorted_probs = sorted_probs / total
                nt = int(sorted_idx[torch.multinomial(sorted_probs, 1)])
            else:
                nt = int(torch.multinomial(probs, 1))
        elif blocked:
            # Walk down the ranked candidates until one clears the n-gram
            # constraint, rather than unconditionally masking (cheap: only
            # triggers once a repeat is actually about to happen).
            ranked = torch.argsort(logits, descending=True)
            nt = next((int(t) for t in ranked.tolist() if t not in blocked),
                      int(ranked[0]))
        else:
            nt = int(logits.argmax(-1))
        loop_sum += es_last + 1
        loop_count += 1
        if nt in eos_ids:
            hit_eos = True
            break
        new_ids.append(nt)
        cur = torch.tensor([[nt]], device=dev)
        cache_position = torch.tensor([L + step], device=dev)
    text = hf_tok.decode(new_ids, skip_special_tokens=True)
    avg_loops = (loop_sum / loop_count) if loop_count else 0.0
    if session is not None:
        session["ids"] = full_ids + new_ids
        session["cache"] = cache
    # hit_eos=False means the loop only stopped because it ran out of
    # max_new budget mid-generation -- for a model that reasons in plain
    # text before its answer (Ouro-1.4B-Thinking never closes </think> until
    # it's actually done), this is the common case at small budgets, and the
    # caller needs to know so it can report finish_reason="length" instead of
    # a misleading "stop" (see chat_completions: agent harnesses like
    # terminal-bench's terminus-2 treat "stop" as "here is the final,
    # complete answer" and never learn a response was silently truncated).
    return text, n_prompt, len(new_ids), avg_loops, hit_eos


# ===========================================================================
# OpenAI-compatible schema (subset: only what a terminus-style agent needs)
# ===========================================================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    # Agent harnesses built against normal chat models (e.g. terminal-bench's
    # terminus-2 / LiteLLM) never send an explicit max_tokens, so THIS default
    # is the entire budget Ouro-1.4B-Thinking gets. Its chat template always
    # opens the assistant turn with a literal '<think>\n' (see generate_chat's
    # enable_thinking comment) and it reliably reasons in plain text for
    # 800-1000+ tokens before ever reaching </think> and its actual answer --
    # confirmed against a full terminal-bench run: every one of 152 sampled
    # completions at the old default of 1024 was cut off mid-reasoning with
    # no </think> and no JSON action, a 0% pass rate purely from truncation,
    # not model capability. 1024 was sized for a non-reasoning chat turn.
    max_tokens: int | None = 8192
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    stream: bool | None = False


app = FastAPI()
_state: dict[str, Any] = {}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(400, "streaming is not supported by this server")
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    max_new = int(req.max_tokens or 8192)

    loop = asyncio.get_running_loop()
    async with _state["lock"]:
        # Single global session: this server is single-request-at-a-time (the
        # lock above), so at most one conversation is ever "in flight" -- the
        # next request either continues it (cache hit, handled in
        # generate_chat) or starts a new one (cache miss -> safe fresh prefill).
        session = _state.setdefault("session", {})
        text, n_prompt, n_completion, avg_loops, hit_eos = await loop.run_in_executor(
            None,
            functools.partial(
                generate_chat,
                _state["model"], _state["tok"], _state["eos_ids"],
                messages, max_new, _state["exit_threshold"],
                _state["no_repeat_ngram_size"], session,
                # Honour the caller's sampling settings. --force_temperature
                # overrides them, which is what makes a pass@k run actually
                # sample k different trajectories even if a harness hardcodes
                # temperature=0.
                (_state["force_temperature"]
                 if _state["force_temperature"] is not None
                 else (req.temperature or 0.0)),
                (req.top_p if req.top_p is not None else 1.0),
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
            # "length" (not "stop") when the loop ran out of max_new budget
            # without hitting EOS -- LiteLLM/terminus-2 raise
            # OutputLengthExceededError on "length" and can recover (e.g.
            # retry/salvage), but silently treating a mid-thought cutoff as a
            # normal "stop" hands the harness a response it can never parse
            # as a valid action with no signal that anything went wrong.
            "finish_reason": "stop" if hit_eos else "length",
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
    ap.add_argument(
        "--no_repeat_ngram_size", type=int, default=3,
        help="Ban a token that would repeat an n-gram already generated this "
        "response (0 disables). Greedy decode has no other defense against "
        "repetition loops on structured/repetitive agent outputs.",
    )
    ap.add_argument(
        "--dtype", default="float32", choices=["float32", "bfloat16"],
        help="fp32 is the eval-script default (bit-exact reproducibility for "
        "benchmark scoring). This server drives interactive agent harnesses "
        "instead, where finishing in reasonable time matters far more than "
        "exact reproducibility -- bf16 is roughly 2x+ faster on this "
        "non-batched, single-request decode loop.",
    )
    ap.add_argument(
        "--force_temperature", type=float, default=None,
        help="Override the temperature every request asks for. Decoding is "
        "greedy at 0, which makes any pass@k evaluation degenerate: k attempts "
        "at one task return byte-identical output, so pass@k == pass@1. Set "
        "this (e.g. 0.7) to make repeated attempts genuinely independent "
        "samples. Left unset, the per-request temperature is used.",
    )
    args = ap.parse_args()

    _dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model, tok, eos_ids = _load_hf_generate_model(
        args.hf_dir, dtype=_dtype, attn_impl=args.attn_impl)
    _state["model"] = model
    _state["tok"] = tok
    _state["eos_ids"] = eos_ids
    _state["model_name"] = args.model_name
    _state["exit_threshold"] = args.early_exit_threshold
    _state["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    _state["force_temperature"] = args.force_temperature
    _state["lock"] = asyncio.Lock()

    print(f"[ouro-server] ready: model={args.model_name} "
          f"exit_threshold={args.early_exit_threshold} "
          f"no_repeat_ngram_size={args.no_repeat_ngram_size} on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
