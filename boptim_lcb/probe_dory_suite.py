"""Probe access to the Dory1.0 suite pieces a 2.6B non-agentic model can run: GPQA Diamond (gated) and IFBench (public)."""
from huggingface_hub import HfApi
api = HfApi()
try:
    print("hf user:", api.whoami().get("name"))
except Exception as e:
    print("hf whoami failed:", type(e).__name__)
for repo in ["Idavidrein/gpqa", "allenai/IFBench_test", "allenai/IFBench_multi-turn", "hendrydong/gpqa_diamond_mc", "fingertap/GPQA-Diamond", "jeggers/gpqa_formatted", "cais/hle", "google/IFEval"]:
    try:
        info = api.dataset_info(repo)
        print("OK   ", repo, "gated=", getattr(info, "gated", None), [s.rfilename for s in info.siblings][:6])
    except Exception as e:
        print("FAIL ", repo, type(e).__name__, str(e)[:90].replace(chr(10), " "))
