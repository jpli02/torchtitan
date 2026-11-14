from huggingface_hub import HfApi, hf_hub_download as d
R = "/home/jli199/boptim_scratch/more_bench_raw/bfcl"
for f in ["BFCL_v3_live_simple.json", "BFCL_v3_live_multiple.json", "BFCL_v3_live_parallel.json", "BFCL_v3_live_parallel_multiple.json"]:
    print(d("gorilla-llm/Berkeley-Function-Calling-Leaderboard", f, repo_type="dataset", local_dir=R))
    print(d("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "possible_answer/" + f, repo_type="dataset", local_dir=R))
api = HfApi()
info = api.dataset_info("HaotongYang/NUPA_text")
files = [s.rfilename for s in info.siblings]
print("NUPA gated=", getattr(info, "gated", None), len(files), "files"); print(files[:60])
