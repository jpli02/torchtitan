from huggingface_hub import HfApi
api = HfApi()
for repo in ["livecodebench/execution-v2", "livecodebench/test_generation", "edinburgh-dawg/mmlu-redux-2.0", "baharef/ToT", "Multilingual-Multimodal-NLP/TableBench", "gorilla-llm/Berkeley-Function-Calling-Leaderboard", "ibm-research/acp_bench", "epfl-dlab/JSONSchemaBench", "madrylab/gsm8k-platinum", "madrylab/platinum-bench"]:
    try:
        info = api.dataset_info(repo)
        files = [s.rfilename for s in info.siblings]
        print("OK  ", repo, "gated=", getattr(info, "gated", None), len(files), "files:", [f for f in files if not f.startswith(".")][:7])
    except Exception as e:
        print("FAIL", repo, type(e).__name__, str(e)[:80].replace(chr(10), " "))
