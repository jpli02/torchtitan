"""Remaining credit of the OpenRouter key in the environment (OPENAI_API_KEY), numbers only: never the key or its label.
Used before a batch of LLM-optimizer runs so that it cannot starve another search that shares the key.
"""
import json
import os
import sys
import urllib.request

key = os.environ.get("OPENAI_API_KEY", "")
if not key:
    print("no key in the environment")
    sys.exit(1)
try:
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    d = json.load(urllib.request.urlopen(req, timeout=30)).get("data", {})
    keep = {k: d.get(k) for k in ("usage", "usage_daily", "usage_monthly", "limit", "limit_remaining", "limit_reset", "is_free_tier", "expires_at")}
    print("key:", json.dumps(keep))
except Exception as e:  # noqa: BLE001
    print(f"key query failed: {type(e).__name__} status={getattr(e, 'code', None)}")
try:
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {key}"})
    d = json.load(urllib.request.urlopen(req, timeout=30)).get("data", {})
    print("account:", json.dumps({k: d.get(k) for k in ("total_credits", "total_usage")}))
except Exception as e:  # noqa: BLE001
    print(f"credits query failed: {type(e).__name__} status={getattr(e, 'code', None)}")
