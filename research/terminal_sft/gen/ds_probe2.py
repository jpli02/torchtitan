import os
for line in open(os.path.expanduser("~/.boptim_keys.env")):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line: continue
    k, v = line.split("=", 1); k = k.replace("export ", "").strip()
    os.environ.setdefault(k, v.strip().strip('"').strip("'"))
os.environ.setdefault("OPENROUTER_API_KEY", os.environ.get("QWEN_API_KEY", ""))
import litellm
litellm.drop_params = True
mid = "openrouter/deepseek/deepseek-v4.1-flash"
msgs = [{"role": "user", "content": "Give a JSON object {\"x\": 1}."}]
# 1. json_object response_format
for rf in [{"type": "json_object"},
           {"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object", "properties": {"x": {"type": "integer"}}}}}]:
    try:
        r = litellm.completion(model=mid, messages=msgs, temperature=0.7, response_format=rf, drop_params=True)
        print("rf", rf.get("type"), "OK:", r.choices[0].message.content[:30])
    except Exception as e:
        print("rf", rf.get("type"), "ERR", type(e).__name__, str(e)[:220])
# 2. big max_tokens (default context) -> 402?
try:
    r = litellm.completion(model=mid, messages=msgs, temperature=0.7, max_tokens=60000, drop_params=True)
    print("max60k OK")
except Exception as e:
    print("max60k ERR", type(e).__name__, str(e)[:220])
