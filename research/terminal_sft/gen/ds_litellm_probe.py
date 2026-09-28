import os, sys
# load keys
for line in open(os.path.expanduser("~/.boptim_keys.env")):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1); k = k.replace("export ", "").strip()
    os.environ.setdefault(k, v.strip().strip('"').strip("'"))
os.environ.setdefault("OPENROUTER_API_KEY", os.environ.get("QWEN_API_KEY", ""))
import litellm
litellm.drop_params = True
for mid in ["openrouter/deepseek/deepseek-v4.1-flash", "openrouter/deepseek/deepseek-chat-v3.1"]:
    try:
        r = litellm.completion(model=mid, messages=[{"role": "user", "content": "Reply OK"}],
                               max_tokens=4000, temperature=0.7)
        print(mid, "OK:", r.choices[0].message.content[:40])
    except Exception as e:
        print(mid, "ERR", type(e).__name__, str(e)[:350])
