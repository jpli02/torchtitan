from huggingface_hub import HfApi, hf_hub_download as d
api = HfApi()
files = [s.rfilename for s in api.dataset_info("gorilla-llm/Berkeley-Function-Calling-Leaderboard").siblings]
print(len(files), "files"); print([f for f in files if "possible" not in f][:40]); print([f for f in files if "possible" in f][:30])
want = [f for f in files if any(k in f for k in ("_simple.json", "_multiple.json", "_parallel.json", "_parallel_multiple.json", "_java.json", "_javascript.json")) and "exec_" not in f and "live" not in f]
R = "/home/jli199/boptim_scratch/more_bench_raw/bfcl"
for f in want:
    print(d("gorilla-llm/Berkeley-Function-Calling-Leaderboard", f, repo_type="dataset", local_dir=R))
