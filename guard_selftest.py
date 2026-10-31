"""Offline unit test for guard_agent helpers (run with the terminal_bench venv)."""
from guard_agent import at_continuation, at_prompt, awaiting_input, unbalanced_quote

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
    "tar -tf archive.tar\narchive/root@f32a54707a4c:/app#": (True, False),
    'echo "Pipeline completed successfully!"root@3eaae8227114:/app# ': (True, False),
}
for s, (p, c) in screens.items():
    gp, gc = at_prompt(s), at_continuation(s)
    flag = "OK " if (gp, gc) == (p, c) else "BAD"
    ok &= (gp, gc) == (p, c)
    print(f"{flag} prompt={gp!s:5} cont={gc!s:5}  {s[-30:]!r}")
inputs = {
    "New Terminal Output:\n\nDo you want to continue? [Y/n]\n": True,
    "Enter password: ": True,
    "Downloading torch...\n 45%|####   | ": False,
    "root@abc:/app# ": False,
    "Collecting numpy\n  Downloading numpy-2.0.whl (16 MB)": False,
}
for s, want in inputs.items():
    got = awaiting_input(s)
    flag = "OK " if got == want else "BAD"
    ok &= got == want
    print(f"{flag} awaiting_input={got!s:5} want {want!s:5}  {s[-32:]!r}")
print("ALL OK" if ok else "FAILURES")
