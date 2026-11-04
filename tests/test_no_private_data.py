"""Guard for the public repo: fails if any tracked file (or the commit author) leaks private data.

Also installed as a git pre-commit hook by scripts/install_hooks.py, so a leak is caught before it
is committed, not after it is public.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATTERNS = {
    "email address": re.compile(r"[\w.+-]+@(?!users\.noreply\.github\.com)[\w-]+\.[\w.]+"),
    "private/LAN IPv4": re.compile(r"\b(10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)\b"),
    "Windows user path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+\w+", re.I),
    "secret/token": re.compile(r"(gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY)"),
}
# the check script itself lists the patterns; real code never needs these literals
SKIP = {"tests/test_no_private_data.py", "scripts/install_hooks.py"}


def _tracked_and_staged():
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    return [f for f in out if f and f not in SKIP]


def test_no_private_data_in_files():
    hits = []
    for f in _tracked_and_staged():
        p = ROOT / f
        if not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue                                     # binary file
        for name, rx in PATTERNS.items():
            for m in rx.finditer(text):
                hits.append(f"{f}: {name}: {m.group(0)[:40]}")
    assert not hits, "private data found:\n" + "\n".join(hits)


def test_git_identity_is_noreply():
    email = subprocess.run(["git", "config", "user.email"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert email.endswith("@users.noreply.github.com"), f"git user.email is {email!r}: use the GitHub noreply address"
