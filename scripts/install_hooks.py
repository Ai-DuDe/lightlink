"""Install a pre-commit hook that runs the privacy test before every commit.  python scripts/install_hooks.py"""
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOK = ROOT / ".git" / "hooks" / "pre-commit"

HOOK.write_text(
    "#!/bin/sh\n"
    "# blocks commits that contain emails, LAN IPs, user paths or tokens\n"
    f'"{sys.executable}" -m pytest -q "{ROOT / "tests" / "test_no_private_data.py"}" || '
    '{ echo "Commit blocked: private data found (see above)."; exit 1; }\n')
HOOK.chmod(HOOK.stat().st_mode | stat.S_IEXEC)
print("installed", HOOK)
