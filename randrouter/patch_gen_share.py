"""One-off edit (2026-11-20): let several teacher-generation processes share the work, and let core.load_teacher read their files.
Kept for the record; running it twice is refused by the asserts."""
import pathlib

here = pathlib.Path(__file__).parent
p = here / "gen_teacher.py"
s = p.read_text()
old = 'ap.add_argument("--shard", default="0/1", help="i/k: this process takes every k-th batch starting at i (several processes, one file each)")\n'
assert s.count(old) == 1
s = s.replace(old, old + 'ap.add_argument("--reverse", action="store_true", help="work through the batches from the end (a second process meeting the first)")\n'
                         'ap.add_argument("--also", default="", help="comma-separated output files of other processes: their problems are skipped too (re-read before every batch)")\n'
                         'ap.add_argument("--mem_gb", type=float, default=0.0, help="cap this process\'s GPU memory (shared card)")\n')
old = '''si, sk = (int(v) for v in args.shard.split("/"))
batches = [b for j, b in enumerate(batches) if j % sk == si]
'''
new = '''if args.reverse:
    batches = batches[::-1]
si, sk = (int(v) for v in args.shard.split("/"))
batches = [b for j, b in enumerate(batches) if j % sk == si]


def others_done():
    ids = set()
    for f in [x for x in args.also.split(",") if x]:
        if os.path.exists(f):
            for l in open(f):
                try:
                    ids.add(json.loads(l)["question_id"])
                except Exception:  # a line being written
                    pass
    return ids


'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''model = AutoModelForCausalLM.from_pretrained(args.ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
'''
new = '''if args.mem_gb:
    torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
model = AutoModelForCausalLM.from_pretrained(args.ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''        batch = [r for r in batch if r["question_id"] not in done]
        if not batch:
            continue
'''
new = '''        skip = done | others_done()
        batch = [r for r in batch if r["question_id"] not in skip]
        if not batch:
            continue
'''
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace('''already in --out are skipped, so the run can be stopped and continued.''',
              '''already in --out are skipped, so the run can be stopped and continued. Several processes can share the work: each writes its
own --out and lists the others' files in --also; a second process started with --reverse meets the first one in the middle.''')
p.write_text(s)

p = here / "core.py"
s = p.read_text()
old = '''    seqs = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if split is not None and r.get("split") != split:
            continue
'''
new = '''    import glob

    files = []
    for part in path.split(","):
        files += sorted(glob.glob(part)) if any(c in part for c in "*?[") else [part]
    seqs, seen = [], set()
    lines = [line for f in files for line in open(f)]
    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:  # a line still being written by a generation process
            continue
        if r["question_id"] in seen:
            continue
        seen.add(r["question_id"])
        if split is not None and r.get("split") != split:
            continue
'''
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace('''    """Rows of gen_teacher.py as {"question_id", "ids", "n_prompt", ...}; sequences longer than max_len are cut at the end."""''',
              '''    """Rows of gen_teacher.py as {"question_id", "ids", "n_prompt", ...}; sequences longer than max_len are cut at the end.
    `path` may list several files (comma-separated, globs allowed); a problem generated twice is taken once."""''')
p.write_text(s)
print("patched gen_teacher.py and core.py")
