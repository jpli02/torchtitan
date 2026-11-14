from huggingface_hub import hf_hub_download as d, snapshot_download as sd
R = "/home/jli199/boptim_scratch/more_bench_raw"
print(d("livecodebench/execution-v2", "data/test-00000-of-00001.parquet", repo_type="dataset", local_dir=R + "/lcb_exec"))
print(d("livecodebench/test_generation", "test.parquet", repo_type="dataset", local_dir=R + "/lcb_testgen"))
print(d("baharef/ToT", "tot_arithmetic/test.json", repo_type="dataset", local_dir=R + "/tot"))
print(d("baharef/ToT", "tot_semantic/test.json", repo_type="dataset", local_dir=R + "/tot"))
print(sd("edinburgh-dawg/mmlu-redux-2.0", repo_type="dataset", local_dir=R + "/mmlu_redux", allow_patterns=["*.arrow"]))
print(d("madrylab/gsm8k-platinum", "main/test-00000-of-00001.parquet", repo_type="dataset", local_dir=R + "/gsm8k_platinum"))
print(d("Multilingual-Multimodal-NLP/TableBench", "TableBench.jsonl", repo_type="dataset", local_dir=R + "/tablebench"))
