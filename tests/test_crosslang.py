"""Compile firmware/lightlink_tx/protocol.h on the PC and check it against the shared vectors.

Compiler search order: $LIGHTLINK_CXX, g++, clang++, then `python -m ziglang c++`.
If none exists the test is SKIPPED (not passed) (see README, Tests).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _compiler():
    env = os.environ.get("LIGHTLINK_CXX")
    if env:
        return env.split()
    for c in ("g++", "clang++"):
        if shutil.which(c):
            return [c]
    try:
        import ziglang  # noqa: F401
        return [sys.executable, "-m", "ziglang", "c++"]
    except ImportError:
        return None


@pytest.fixture(scope="module")
def dump_exe(tmp_path_factory):
    cxx = _compiler()
    if cxx is None:
        pytest.skip("no C++ compiler (set LIGHTLINK_CXX, install g++, or pip install ziglang)")
    exe = tmp_path_factory.mktemp("cxx") / ("dump.exe" if os.name == "nt" else "dump")
    src = ROOT / "firmware" / "tests" / "dump_frames.cpp"
    r = subprocess.run(cxx + ["-std=c++17", "-Wall", "-Werror", "-o", str(exe), str(src)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def test_cpp_matches_python_vectors(dump_exe):
    data = json.loads((ROOT / "protocol" / "vectors.json").read_text())
    vecs = data["vectors"]
    stdin = "\n".join(v["payload_hex"] or "-" for v in vecs) + "\n"
    out = subprocess.run([str(dump_exe)], input=stdin, capture_output=True, text=True, check=True)
    lines = out.stdout.strip().splitlines()
    assert lines[-1] == f"crc {data['crc8_check']['crc']:02x}"
    for v, line in zip(vecs, lines):
        frame_hex, symbols = line.split()
        assert frame_hex == v["frame_hex"], v["payload_hex"]
        assert symbols == v["symbols"], v["payload_hex"]
