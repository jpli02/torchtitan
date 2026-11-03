"""Semantic Scholar citation counts (closer to Google Scholar than OpenAlex) for the papers listed in
bench_cites.py, one request per paper with slow pacing and long backoff on HTTP 429.
Usage: python bench_cites_s2.py OUT.tsv"""
import json
import os
import sys
import time
import urllib.request

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_cites.py")).read()
ns = {}
exec(src.split("def get(")[0], ns)  # noqa: S102 - our own file; only the constants section (PAPERS list)
PAPERS = ns["PAPERS"]


def get(url, tries=8):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bench-survey/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa
            wait = 15 * (t + 1)
            print(f"retry {t} ({e}); sleeping {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    return None


rows = []
for label, arx, dom, note in PAPERS:
    r = get(f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arx}?fields=title,year,venue,citationCount")
    time.sleep(1.5)
    if r is None:
        rows.append((label, arx, dom, note, None, None, "", ""))
    else:
        rows.append((label, arx, dom, note, r.get("citationCount"), r.get("year"), (r.get("venue") or "")[:40], (r.get("title") or "")[:70]))
    print(rows[-1][:6], flush=True)
rows.sort(key=lambda x: -(x[4] or 0))
with open(sys.argv[1], "w") as f:
    f.write("label\tarxiv\tdomain\tnote\tcites\tyear\tvenue\ttitle\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print("DONE")
for r in rows:
    print(f"{str(r[4]):>6}  {str(r[5]):>4}  {r[0]:<30} {r[2]:<12} {r[6]:<28} {r[3]}")
