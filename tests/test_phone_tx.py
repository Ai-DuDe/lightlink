"""The phone page's JavaScript codec must produce the same frames as the Python one (needs Node)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_codec_matches_vectors():
    data = json.loads((ROOT / "protocol" / "vectors.json").read_text())
    # the codec lives inside an HTML <script>; extract it so Node can run it unchanged
    html = (ROOT / "docs" / "index.html").read_text()
    code = html.split("<script>")[1].split("</script>")[0]
    runner = code + f"""
const vecs = {json.dumps(data['vectors'])};
const out = vecs.map(v => {{
  const p = Array.from(Buffer.from(v.payload_hex, 'hex'));
  const f = buildFrame(p);
  return [Buffer.from(f).toString('hex'), manchester(f).join('')];
}});
console.log(JSON.stringify(out));
"""
    r = subprocess.run(["node", "-e", runner], capture_output=True, text=True, check=True)
    for v, (fh, sy) in zip(data["vectors"], json.loads(r.stdout)):
        assert fh == v["frame_hex"] and sy == v["symbols"]
