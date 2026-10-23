import os, json, urllib.request, urllib.error
for line in open(os.path.expanduser("~/.boptim_keys.env")):
    line = line.strip()
    if line.startswith("export "): line = line[7:]
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("QWEN_API_KEY")
body = json.dumps({"model": "deepseek/deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "Reply with the single word OK."}],
    "max_tokens": 50, "temperature": 0}).encode()
req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode()
    print("RAW RESPONSE:", raw[:900])
except urllib.error.HTTPError as e:
    print("HTTPError", e.code, e.reason, "->", e.read().decode()[:300])
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:200])
