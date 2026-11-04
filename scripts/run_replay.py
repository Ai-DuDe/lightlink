"""Run the full receiver with no hardware.

  python scripts/run_replay.py --simulate [--bit-rate 8] [--noise 6] [--block 15 25] [--no-window]
  python scripts/run_replay.py --video logs/capture_xxx.avi
  python scripts/run_replay.py --simulate --save-sim sim.avi     (write the simulated clip)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lightlink.app import run_dashboard  # noqa: E402
from lightlink.config import load_config  # noqa: E402
from lightlink.replay import read_video, run_frames, save_video  # noqa: E402
from lightlink.simulator import Simulator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--video")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--bit-rate", type=float)
    ap.add_argument("--noise", type=float)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--payload")
    ap.add_argument("--block", nargs=2, type=float, action="append", metavar=("T0", "T1"))
    ap.add_argument("--save-sim")
    ap.add_argument("--no-window", action="store_true", help="print results only")
    a = ap.parse_args()
    if not a.video and not a.simulate:
        ap.error("choose --video FILE or --simulate")
    sim, proto = {}, {}
    if a.noise is not None: sim["noise_sigma"] = a.noise
    if a.duration: sim["duration_s"] = a.duration
    if a.payload: sim["payload"] = a.payload
    if a.block: sim["blocks"] = a.block
    if a.bit_rate: proto["bit_rate_bps"] = a.bit_rate
    cfg = load_config(a.config, {"sim": sim, "protocol": proto})

    if a.simulate:
        s = Simulator(cfg.sim, cfg.protocol.bit_rate_bps, cfg.protocol.gap_s)
        if a.save_sim:
            save_video(a.save_sim, s.frames(), cfg.sim.fps)
            print("saved", a.save_sim)
            return
        source = ((f.t, f.image) for f in s.frames())
    else:
        source = read_video(a.video)

    if a.no_window:
        res = run_frames(cfg, source)
        for ev in res.events:
            print(f"{ev.t_start:7.2f}s  {'VALID ' + ev.payload.decode('utf-8', 'replace') if ev.valid else 'FAILED ' + ev.reason}")
        print(f"{res.frames} frames, {sum(e.valid for e in res.events)} valid, "
              f"{sum(not e.valid for e in res.events)} failed, first LOCKED at {res.first_time('LOCKED')}")
    else:
        run_dashboard(cfg, source, realtime=True)


if __name__ == "__main__":
    main()
