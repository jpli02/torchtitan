"""One small chat call to check that the key and the model name work before a batch of LLM-optimizer runs is started
(randrouter/opt_compare_llm.sh). Reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment and prints only "ok" or the
error's class and status code: never the key, never the response headers.
"""
import os
import sys

import openai

model = os.environ.get("PING_MODEL", "openai/gpt-6-luna")
try:
    r = openai.OpenAI().chat.completions.create(model=model, messages=[{"role": "user", "content": "Reply with the word ok."}],
                                               max_completion_tokens=2000)
    u = getattr(r, "usage", None)
    print(f"LLM ping ok: {model}" + ("" if u is None else f" (prompt {u.prompt_tokens}, completion {u.completion_tokens} tokens)"))
except Exception as e:  # noqa: BLE001
    print(f"LLM ping failed: {type(e).__name__} status={getattr(e, 'status_code', None)}")
    sys.exit(1)
