import re
src = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/boptim-experiments.html"
out = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/boptim-experiments-pub.html"
t = open(src).read()
i = t.find("<body>")
if i != -1:
    t = t[i + len("<body>"):]
t = re.sub(r"\s*</body>\s*</html>\s*$", "\n", t)
t = t.lstrip("\n")
open(out, "w").write(t)
print("first 60:", repr(t[:60]))
print("last 40:", repr(t[-40:]))
print("has doctype:", "<!doctype" in t.lower(), "has <head>:", "<head>" in t.lower(), "body-open:", "<body>" in t.lower())
print("has new Q7:", "Does matching the data distribution per skill" in t)
print("has 2.6B:", "Ouro-2.6B-Thinking" in t)
