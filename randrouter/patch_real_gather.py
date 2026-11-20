"""One-off edit (2026-11-20) of lcb_real_eval2.py: --gather runs the router in "gather" mode (depth_eval --threshold_steps): every
token computes all four loops, so the cache is exact, and only its readout is taken from the loop where the router would have
let it leave. No compute is saved; it is the real-execution upper bound of any cache repair for that router ("what would this
router score if the skipped loops' keys and values were right"). The record then reports the loops the router WOULD execute.
Kept for the record; the asserts refuse a second run."""
import pathlib

p = pathlib.Path(__file__).parent / "lcb_real_eval2.py"
s = p.read_text()
old = 'ap.add_argument("--fallback_free_mb", type=int, default=12000)\n'
assert s.count(old) == 1
s = s.replace(old, old + 'ap.add_argument("--gather", action="store_true", help="exact cache: all loops computed, readout at the router\'s exit loop (upper bound of a cache repair)")\n')
old = '''           "--cfg", "decode_early_exit_threshold=" + json.dumps(thr), "--out", str(out)]
'''
new = '''           "--out", str(out)]
    if args.gather:
        cmd += ["--threshold_steps", ",".join(str(v) for v in thr[:3])]
    else:
        cmd += ["--cfg", "decode_early_exit_threshold=" + json.dumps(thr)]
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''exec_loops = (sum(r["exec_loops"] * r["exec_n"] for r in rs if r.get("exec_n")) / ex_n) if ex_n else 4.0
'''
new = '''exec_loops = (sum(r["exec_loops"] * r["exec_n"] for r in rs if r.get("exec_n")) / ex_n) if ex_n else 4.0
# gather mode: the loops the router would execute (the stock selection's counter), weighted by generated tokens
would_loops = (sum(r["mean_loops"] * r["mean_tokens"] * r["n"] for r in rs) / max(sum(r["mean_tokens"] * r["n"] for r in rs), 1e-9)) if args.gather else None
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''rec = {"ckpt": args.ckpt, "thresholds": thr, "cfg": args.cfg, "n": n_tot,'''
assert s.count(old) == 1
s = s.replace(old, '''rec = {"ckpt": args.ckpt, "thresholds": thr, "cfg": args.cfg, "gather": bool(args.gather), "would_loops": would_loops, "n": n_tot,''')
p.write_text(s)
print("patched lcb_real_eval2.py")
