"""Offline unit test for guard_agent helpers (run with the terminal_bench venv)."""
from guard_agent import at_continuation, at_prompt, unbalanced_quote

tests = {
    'aws s3api put-bucket-acl --bucket b --policy "{\\"Version\\": \\"2012\\"}"\n': None,
    'aws s3api x --policy "{\\"Version\\": \\"2012}\n': '"',
    "cat > f.py << 'EOF'\nprint(\"it's\")\nEOF\n": None,
    "grep '#' file.txt\n": None,
    'echo "a\\"b"\n': None,
    "echo 'unterminated\n": "'",
    "python3 -c 'import json; print(json.dumps({\"a\": 1}))'\n": None,
    "# comment with ' apostrophe\nls\n": None,
}
ok = True
for k, want in tests.items():
    got = unbalanced_quote(k)
    flag = "OK " if got == want else "BAD"
    ok &= got == want
    print(f"{flag} {got!r:5} want {want!r:5}  {k[:60]!r}")
screens = {
    "root@abc123:/app# ": (True, False),
    "root@abc123:/app# ls\nfile.txt\nroot@abc123:/app# ": (True, False),
    "aws s3api list-buckets\n> ": (False, True),
    "> aws s3api list-buckets\n> ": (False, True),
    "Downloading torch...\n 45%|####   | ": (False, False),
    "$ ": (True, False),
    "dquote> ": (False, True),
}
for s, (p, c) in screens.items():
    gp, gc = at_prompt(s), at_continuation(s)
    flag = "OK " if (gp, gc) == (p, c) else "BAD"
    ok &= (gp, gc) == (p, c)
    print(f"{flag} prompt={gp!s:5} cont={gc!s:5}  {s[-30:]!r}")
print("ALL OK" if ok else "FAILURES")
