"""Wire page_blocks_1107 into page_update_1102.py (idempotent): import, BO block after the LCB wall-clock table, Dory-suite block after the
selected-benchmark table, token-distribution block after the ep4 training metrics."""
p = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/page_update_1102.py"
s = open(p).read()
if "page_blocks_1107" in s:
    print("already patched")
    raise SystemExit(0)
reps = [
    ("import json\nimport os\nimport shutil\nimport sys\n", "import json\nimport os\nimport shutil\nimport sys\n\nimport page_blocks_1107 as pb  # 2026-11-07 blocks: token distribution, Dory suite, LCB BO\n"),
    ("    bcb_block + lcb_block + lcb_wc_block + gsm_wc_block + codepro_wc_block + ", "    bcb_block + lcb_block + lcb_wc_block + pb.lcb_bo_block() + gsm_wc_block + codepro_wc_block + "),
    ("cite_block.rstrip(\"\\n\") + (\"\\n\" + more_block.rstrip(\"\\n\") if more_block else \"\"))",
     "cite_block.rstrip(\"\\n\") + \"\\n\" + pb.dory_block().rstrip(\"\\n\") + (\"\\n\" + more_block.rstrip(\"\\n\") if more_block else \"\"))"),
    ("{ep4_metrics}<div class=\"sub-head\"><h3>ep4 and r1 pass@1 on their training set</h3>", "{ep4_metrics}{pb.tokdist_block()}<div class=\"sub-head\"><h3>ep4 and r1 pass@1 on their training set</h3>"),
]
for old, new in reps:
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched page_update_1102.py")
