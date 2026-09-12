#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/chat/completions server for a local HF Qwen
checkpoint (the SFT'd BO agent), so boptim-agent's `qwen` optimizer can talk
to it unchanged:

  QWEN_BACKEND=api QWEN_BASE_URL=http://127.0.0.1:8210/v1 QWEN_API_KEY=local \
  python main.py --optim qwen --model qwen-bo ...

No vLLM on this box; plain transformers generate, one request at a time.
The optimizer sends [system, user] and parses JSON after stripping <think>
blocks, so we render with the tokenizer's chat template as-is (thinking left
to the model, which the SFT taught to emit an empty think block then JSON).

  python qwen_bo_server.py --hf_dir <serving dir> --port 8210 --model_name qwen-bo
"""
import argparse
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
_lock = threading.Lock()
hf_tok = None
hf_model = None
MODEL_NAME = "qwen-bo"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = 4096
    temperature: float | None = 0.7
    top_p: float | None = 0.8
    stream: bool | None = False


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = hf_tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = hf_tok(prompt, return_tensors="pt", add_special_tokens=False).to(hf_model.device)
    max_new = int(min(req.max_tokens or 4096, 8192))
    temp = float(req.temperature if req.temperature is not None else 0.7)
    t0 = time.time()
    with _lock, torch.no_grad():
        out = hf_model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=temp > 0,
            temperature=max(temp, 1e-5) if temp > 0 else None,
            top_p=req.top_p if temp > 0 else None,
            eos_token_id=[hf_tok.eos_token_id, hf_tok.convert_tokens_to_ids("<|im_end|>")],
            pad_token_id=hf_tok.pad_token_id or hf_tok.eos_token_id,
        )
    gen = out[0, enc.input_ids.shape[1]:]
    text = hf_tok.decode(gen, skip_special_tokens=True)
    finish = "length" if len(gen) >= max_new else "stop"
    print(f"[qwen-bo-server] prompt={enc.input_ids.shape[1]} gen={len(gen)} "
          f"{time.time() - t0:.1f}s finish={finish} tail={text[-120:]!r}", flush=True)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": int(enc.input_ids.shape[1]), "completion_tokens": int(len(gen)),
                  "total_tokens": int(enc.input_ids.shape[1] + len(gen))},
    }


def main():
    global hf_tok, hf_model, MODEL_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dir", required=True)
    ap.add_argument("--port", type=int, default=8210)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model_name", default="qwen-bo")
    a = ap.parse_args()
    MODEL_NAME = a.model_name
    hf_tok = AutoTokenizer.from_pretrained(a.hf_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(
        a.hf_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    print(f"[qwen-bo-server] loaded {a.hf_dir} on {hf_model.device}; serving {MODEL_NAME} on :{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
