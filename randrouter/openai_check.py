"""Check the OpenAI key in ~/.boptim_keys.env against api.openai.com (not OpenRouter): number of models visible, whether a few
ids exist, and one small completion. Prints no key material and no response headers.

  /home/jli199/boptim-agent/.venv/bin/python -I randrouter/openai_check.py
"""
import os
import re

import openai

env = open(os.path.expanduser("~/.boptim_keys.env")).read()
key = re.search(r"^export OPENAI_API_KEY=(.*)$", env, flags=re.M).group(1).strip().strip('"').strip("'")
client = openai.OpenAI(api_key=key, base_url="https://api.openai.com/v1")
try:
    ids = sorted(m.id for m in client.models.list())
    print(f"models visible: {len(ids)}")
    for want in ("gpt-6-luna", "gpt-6-luna-pro", "gpt-6-sol", "gpt-6.1-sol", "gpt-6-astra", "gpt-5-mini"):
        print(f"  {want}: {'yes' if want in ids else 'no'}")
    print("  gpt-6 family:", [i for i in ids if i.startswith("gpt-6")][:20])
except Exception as e:  # noqa: BLE001
    print(f"model list failed: {type(e).__name__} status={getattr(e, 'status_code', None)}")
for model in ("gpt-6-luna", "gpt-5-mini"):
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Reply with the word ok."}], max_completion_tokens=2000)
        u = r.usage
        print(f"completion with {model}: ok (prompt {u.prompt_tokens}, completion {u.completion_tokens} tokens): {r.choices[0].message.content[:20]!r}")
        break
    except Exception as e:  # noqa: BLE001
        body = getattr(e, "body", None)
        code = body.get("code") if isinstance(body, dict) else None
        print(f"completion with {model} failed: {type(e).__name__} status={getattr(e, 'status_code', None)} code={code}")
