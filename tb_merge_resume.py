"""Fold the finished tasks of a run's resume root into its main root, so that a further resume (which clears the resume root at
launch) cannot delete them:
  python tb_merge_resume.py ROOT NAME
A task directory moves from ROOT/resume_NAME/NAME/<task> to ROOT/NAME/<task> when it holds a trial results.json and the main root has
no result for that task (an unfinished main directory of the task is set aside as <task>.unfinished)."""
import glob
import os
import shutil
import sys

root, name = sys.argv[1], sys.argv[2]
main, res = f"{root}/{name}", f"{root}/resume_{name}/{name}"
moved = 0
os.makedirs(main, exist_ok=True)
for d in sorted(glob.glob(f"{res}/*/")):
    task = os.path.basename(d.rstrip("/"))
    if not glob.glob(f"{d}*/results.json"):
        continue
    dst = f"{main}/{task}"
    if glob.glob(f"{dst}/*/results.json"):
        continue
    if os.path.exists(dst):
        shutil.move(dst, dst + ".unfinished")
    shutil.move(d.rstrip("/"), dst)
    moved += 1
print(f"{name}: {moved} finished tasks moved from the resume root into the main root")
