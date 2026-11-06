"""Citation counts for benchmark papers from the Semantic Scholar graph API (public, no key).
Usage: python bench_cites.py OUT.tsv   -> table sorted by citations (label, arXiv, domain, note, cites, year, venue)."""
import json
import sys
import time
import urllib.parse
import urllib.request

FIELDS = "title,year,venue,citationCount,externalIds"
# (label, arXiv id, domain, note)
PAPERS = [
    ("LiveCodeBench", "2403.07974", "code", "ICLR 2025; contest problems, tests; v5/v6 (2025 problems)"),
    ("BigCodeBench", "2406.15877", "code", "ICLR 2025; 1,140 lib-call tasks + Hard 148"),
    ("CRUXEval", "2401.03065", "code", "ICML 2024; output prediction"),
    ("SWE-bench", "2310.06770", "code-agent", "ICLR 2024; Verified subset = OpenAI blog 2024"),
    ("SWE-bench Multimodal", "2410.03859", "code-agent", "ICLR 2025"),
    ("SWE-bench Pro", "2509.16941", "code-agent", "2025"),
    ("Multi-SWE-bench", "2504.02605", "code-agent", "2025"),
    ("SWE-bench Live", "2505.23419", "code-agent", "2025"),
    ("LiveCodeBench Pro", "2506.11928", "code", "2025; olympiad-level"),
    ("CodeElo", "2501.01257", "code", "2025; Codeforces Elo"),
    ("KernelBench", "2502.10517", "code", "ICML 2025; CUDA kernels"),
    ("CodeMMLU", "2410.01999", "code-mc", "ICLR 2025; multiple choice code understanding"),
    ("HumanEval Pro / MBPP Pro", "2412.21199", "code", "2025; self-invoking"),
    ("LeetCodeDataset", "2504.14655", "code", "2025"),
    ("SciCode", "2407.13168", "code", "NeurIPS 2024; research coding"),
    ("FullStackBench", "2412.00535", "code", "2024/2025"),
    ("EvalPlus (HumanEval+/MBPP+)", "2305.01210", "code", "NeurIPS 2023"),
    ("HumanEval (Codex)", "2107.03374", "code", "2021"),
    ("MBPP", "2108.07732", "code", "2021"),
    ("DS-1000", "2211.11501", "code", "ICML 2023"),
    ("MHPP", "2405.11430", "code", "2024"),
    ("BigO(Bench)", "2503.15242", "code", "2025; complexity"),
    ("OJBench", "2506.16395", "code", "2025"),
    ("USACO bench", "2404.10952", "code", "2024"),
    ("NaturalCodeBench", "2405.04520", "code", "ACL 2024"),
    ("EvoEval", "2403.19114", "code", "COLM 2024"),
    ("CRUXEval-X", "2408.13001", "code", "2024"),
    ("Spider 2.0", "2411.07763", "code-sql", "ICLR 2025"),
    ("LiveBench", "2406.19314", "mixed", "ICLR 2025; coding+math+reasoning+data, exact-match"),
    ("GSM-Symbolic", "2410.05229", "math", "ICLR 2025"),
    ("MMLU-Pro", "2406.01574", "knowledge-mc", "NeurIPS 2024"),
    ("SuperGPQA", "2502.14739", "knowledge-mc", "2025"),
    ("Humanity's Last Exam", "2501.14249", "knowledge", "2025"),
    ("GPQA", "2311.12022", "science-mc", "COLM 2024"),
    ("ZebraLogic", "2502.01100", "logic", "ICML 2025; exact-match grid puzzles, graded sizes"),
    ("BIG-Bench Extra Hard", "2502.19187", "reasoning", "2025"),
    ("Omni-MATH", "2410.07985", "math", "ICLR 2025; olympiad"),
    ("FrontierMath", "2411.04872", "math", "2024/2025"),
    ("MathArena", "2505.23281", "math", "2025"),
    ("LiveMathBench", "2412.13147", "math", "2025"),
    ("MATH-500 (PRM800K paper)", "2305.20050", "math", "ICLR 2024"),
    ("GSM8K", "2110.14168", "math", "2021"),
    ("MATH", "2103.03874", "math", "NeurIPS 2021"),
    ("ARC-AGI-2", "2505.11831", "reasoning", "2025"),
    ("tau-bench", "2406.12045", "agent", "2024"),
    ("BrowseComp", "2504.12516", "agent", "2025"),
    ("SimpleQA", "2411.04368", "factuality", "2024"),
    ("Terminal-Bench", "2601.11868", "code-agent", "2026 paper; benchmark released 2025"),
    ("AetherCode", "2508.16914", "code", "2025"),
    ("Omni-MATH", "2410.07985", "math", "ICLR 2025; olympiad"),
    ("MATH-Perturb", "2502.06453", "math", "ICML 2025"),
    ("GSM8K-Platinum (reliability)", "2502.03461", "math", "ICML 2025"),
    ("Global-MMLU", "2412.03304", "knowledge-mc", "ACL 2025"),
    ("MMLU-CF", "2412.15194", "knowledge-mc", "ACL 2025"),
    ("IFBench", "2507.02833", "instruction", "2025"),
    ("BBEH", "2502.19187", "reasoning", "2025"),
    ("MathArena", "2505.23281", "math", "2025"),
    ("AIME (MathArena/HMMT)", "2505.23281", "math", "2025"),
    ("LiveBench", "2406.19314", "mixed", "ICLR 2025"),
    ("MedXpertQA", "2501.18362", "medical-mc", "ICML 2025"),
    ("PHYBench", "2504.16074", "physics", "2025"),
    ("SimpleBench", "2410.00134", "reasoning", "2024"),
    ("MMLU-Pro", "2406.01574", "knowledge-mc", "NeurIPS 2024"),
]
SEARCH = ["Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces", "SWE-bench Verified"]


def get(url, data=None, tries=6):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": "bench-survey/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa
            print(f"retry {t}: {e}", file=sys.stderr)
            time.sleep(3 + 5 * t)
    return None


# OpenAlex (polite pool, no key): arXiv preprints are indexed under DOI 10.48550/arXiv.<id>; cited_by_count is
# OpenAlex's count (systematically below Google Scholar, but a consistent ranking).
MAILTO = "jplee0202@gmail.com"
rows = []
for label, arx, dom, note in PAPERS:
    url = f"https://api.openalex.org/works/https://doi.org/10.48550/arXiv.{arx}?mailto={MAILTO}&select=title,publication_year,cited_by_count,primary_location"
    r = get(url, tries=2)
    time.sleep(0.15)
    if r is None:
        rows.append((label, arx, dom, note, None, None, "", ""))
        continue
    venue = ((r.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    rows.append((label, arx, dom, note, r.get("cited_by_count"), r.get("publication_year"), venue[:40], (r.get("title") or "")[:70]))
for q in SEARCH:
    r = get("https://api.openalex.org/works?search=" + urllib.parse.quote(q) + f"&per-page=3&mailto={MAILTO}&select=title,publication_year,cited_by_count,ids", tries=2)
    for d in (r or {}).get("results", []):
        rows.append((f"search:{q[:30]}", str((d.get("ids") or {}).get("doi", "-"))[-30:], "-", "-", d.get("cited_by_count"), d.get("publication_year"), "", (d.get("title") or "")[:70]))
rows.sort(key=lambda x: -(x[4] or 0))
with open(sys.argv[1], "w") as f:
    f.write("label\tarxiv\tdomain\tnote\tcites\tyear\tvenue\ttitle\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
for r in rows:
    print(f"{str(r[4]):>6}  {str(r[5]):>4}  {r[0]:<30} {r[2]:<12} {r[6]:<28} {r[3]}")
