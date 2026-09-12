#!/usr/bin/env python3
"""Show OpenRouter key limits/usage and the full error of one small chat call.
Never prints the key."""
import json
import os
import sys
import urllib.request

sys.path.insert(0, "/home/jli199/boptim-agent")
from gen_bo_traj import _load_keys  # noqa: E402

_load_keys()
base = os.environ["QWEN_BASE_URL"].rstrip("/")
hdr = {"Authorization": "Bearer " + os.environ["QWEN_API_KEY"], "Content-Type": "application/json"}
for path in ("/auth/key", "/credits"):
    try:
        r = urllib.request.urlopen(urllib.request.Request(base + path, headers=hdr), timeout=30)
        d = json.load(r).get("data", {})
        d = {k: v for k, v in d.items() if k not in ("label", "key", "hash")}
        print(path, json.dumps(d))
    except Exception as e:
        print(path, "ERR", str(e)[:200])
model = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen3.7-plus"
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512
body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Say OK."}],
                   "max_tokens": max_tokens}).encode()
try:
    r = urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=body, headers=hdr), timeout=120)
    d = json.load(r)
    print("chat OK:", repr(d["choices"][0]["message"]["content"][:80]), d.get("usage"))
except urllib.error.HTTPError as e:
    print("chat HTTP", e.code, e.read().decode()[:600])
