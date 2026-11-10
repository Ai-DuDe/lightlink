"""RUN THIS. Live receiver: webcam -> dashboard.   python run.py [--bit-rate 4] [--camera 0]

The bit rate must match the sender (default: protocol.bit_rate_bps in config.yaml)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lightlink.app import run_dashboard  # noqa: E402
from lightlink.camera import Camera  # noqa: E402
from lightlink.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--bit-rate", type=float, help="must match the transmitter")
    a = ap.parse_args()
    over = {}
    if a.camera is not None:
        over["camera"] = {"index": a.camera}
    if a.bit_rate is not None:
        over["protocol"] = {"bit_rate_bps": a.bit_rate}
    cfg = load_config(a.config, over)
    cam = Camera(cfg.camera).open()
    r = cam.report
    print(f"camera backend: {r.backend}")
    for k in r.readback:
        print(f"  {k:14s} requested={r.requested[k]} read-back={r.readback[k]} obeyed={r.obeyed[k]}")
    print("exposure locked:", {True: "yes", False: "NO (see README)", None: "unknown"}[r.exposure_locked])
    try:
        run_dashboard(cfg, cam.frames())
    finally:
        cam.close()


if __name__ == "__main__":
    main()
