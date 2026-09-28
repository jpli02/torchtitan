import os, json, urllib.request
for line in open(os.path.expanduser("~/.boptim_keys.env")):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line: continue
    k, v = line.split("=", 1); k = k.replace("export ", "").strip()
    os.environ[k] = v.strip().strip('"').strip("'")
key = os.environ["OPENROUTER_API_KEY"]
base = "https://openrouter.ai/api/v1"
hdr = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
r = urllib.request.urlopen(urllib.request.Request(base + "/credits", headers=hdr), timeout=30)
d = json.load(r).get("data", {})
print("credits: total", d.get("total_credits"), "used", round(d.get("total_usage", 0), 4))
# big-request affordability test (mimics an agent turn with headroom)
body = json.dumps({"model": "deepseek/deepseek-v4.1-flash",
                   "messages": [{"role": "user", "content": "Reply with a JSON object {\"ok\": true} only."}],
                   "max_tokens": 32000}).encode()
try:
    r = urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=body, headers=hdr), timeout=120)
    j = json.load(r)
    print("flash 32k call OK:", repr(j["choices"][0]["message"]["content"][:60]), "cost", j.get("usage", {}).get("cost"))
except urllib.error.HTTPError as e:
    print("flash HTTP", e.code, e.read().decode()[:200])
