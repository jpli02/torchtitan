import os, json, glob
for line in open(os.path.expanduser("~/.boptim_keys.env")):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line: continue
    k, v = line.split("=", 1); k = k.replace("export ", "").strip()
    os.environ.setdefault(k, v.strip().strip('"').strip("'"))
os.environ.setdefault("OPENROUTER_API_KEY", os.environ.get("QWEN_API_KEY", ""))
import litellm
# replay the exact messages the agent sent
f = sorted(glob.glob("/tmp/tb_runs/ds_smoke/*/*/agent-logs/episode-0/debug.json"))[0]
d = json.load(open(f))
msgs = d["messages"]
print("prompt chars:", sum(len(m["content"]) for m in msgs), "max_tokens:", d.get("max_tokens"))
try:
    r = litellm.completion(model="openrouter/deepseek/deepseek-chat-v3-0324",
                           messages=msgs, temperature=0.7, max_tokens=8192, drop_params=True,
                           api_base="https://openrouter.ai/api/v1/chat/completions")
    c = r.choices[0]
    print("finish_reason:", c.finish_reason, "| content chars:", len(c.message.content or ""))
    print("content head:", (c.message.content or "")[:300])
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:500])
