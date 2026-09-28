import json, sys

def load(f):
    d = json.load(open(f)); e = d.get('eval', d)
    out = {}
    for tid, runs in e.items():
        r = runs[0] if isinstance(runs, list) else runs
        st = r.get('base_status') or (r.get('base', [None])[0] if isinstance(r.get('base'), list) else r.get('base'))
        out[tid] = (st == 'pass')
    return out

fp32 = load('outputs/he_fp32_local/samples-sanitized_eval_results.json')
cplx = load('outputs/he_complexsys_local/samples-sanitized_eval_results.json')
keys = sorted(set(fp32) | set(cplx), key=lambda x: int(x.split('/')[1]))

def num(k): return k.split('/')[1]
fp32_pass = sum(fp32.get(k, False) for k in keys)
cplx_pass = sum(cplx.get(k, False) for k in keys)
gained = [num(k) for k in keys if cplx.get(k) and not fp32.get(k)]   # cplx fixed
lost   = [num(k) for k in keys if fp32.get(k) and not cplx.get(k)]   # cplx broke
both_fail = [num(k) for k in keys if not cplx.get(k) and not fp32.get(k)]

print(f"fp32       : {fp32_pass}/164")
print(f"complexsys : {cplx_pass}/164   (net {cplx_pass-fp32_pass:+d})")
print(f"\ncomplexsys GAINED (fail->pass), {len(gained)}: {gained}")
print(f"complexsys LOST   (pass->fail), {len(lost)}: {lost}")
print(f"\nstill failing in BOTH ({len(both_fail)}): {both_fail}")
