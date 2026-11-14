"""Write a standalone preview page of the two BO Pareto charts (light or dark) for a headless-browser screenshot.
  python boptim_lcb/render_pareto_preview.py OUT.html [dark]"""
import sys

sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
import page_blocks_1114 as pb4  # noqa: E402

dark = len(sys.argv) > 2 and sys.argv[2] == "dark"
light_vars = "--bg:#f5f6f8;--surface:#ffffff;--line:#dcdfe6;--line-2:#c6cad3;--text:#14171c;--text-2:#5b6270;--text-3:#8a909c;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--base:#7b8290"
dark_vars = "--bg:#15171b;--surface:#1d2025;--line:#2e323a;--line-2:#3d424c;--text:#f0f1f4;--text-2:#a6acb8;--text-3:#767d8a;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--base:#8e95a3"
css = (f"<style>:root{{{dark_vars if dark else light_vars}}}body{{background:var(--surface);color:var(--text);font-family:system-ui,sans-serif;margin:20px;max-width:800px}}"
       ".src{color:var(--text-2);font-size:13.5px}h3{font-size:18px}</style>")
open(sys.argv[1], "w").write("<!doctype html><meta charset=utf-8>" + css + pb4.pareto_block())
print("wrote", sys.argv[1])
