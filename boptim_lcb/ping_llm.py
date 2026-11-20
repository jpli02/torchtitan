"""Check that the optimizer's LLM answers (same client and call shape as boptim-agent optim/chatgpt.py). Never prints the key.

  OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=... python ping_llm.py openai/gpt-5.6-luna
"""
import os
import sys

import openai

model = sys.argv[1]
client = openai.OpenAI()
try:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are an expert optimization assistant."},
                  {"role": "user", "content": 'Reply with the JSON {"ok": true} and nothing else.'}],
        max_completion_tokens=int(os.environ.get("GPT_MAX_TOKENS", "2000")),
    )
    c = r.choices[0]
    print("model answered:", repr((c.message.content or "")[:80]), "| finish:", c.finish_reason, "| served by:", getattr(r, "model", "?"),
          "| tokens:", getattr(r.usage, "total_tokens", "?"))
except Exception as e:  # noqa: BLE001
    msg = str(e)
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        msg = msg.replace(key, "<key>")
    print("PING FAILED:", type(e).__name__, msg[:300])
    sys.exit(1)
