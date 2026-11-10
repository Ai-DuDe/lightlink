"""Is this camera good enough?   python scripts/camera_check.py [--camera 0] [--seconds 8] [--cover-test]

Measures true FPS, exposure-lock behaviour and clipping, then says in plain words whether
the camera suits the link and what bit rate to use.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lightlink.camera import Camera, fps_stats, max_bit_rate  # noqa: E402
from lightlink.config import load_config  # noqa: E402


def collect(cam, seconds):
    stamps, means, clip_hi, clip_lo = [], [], [], []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        r = cam.read()
        if r is None:
            break
        t, f = r
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        stamps.append(t)
        means.append(float(g.mean()))
        clip_hi.append(float((g >= 250).mean()))
        clip_lo.append(float((g <= 5).mean()))
    return stamps, np.array(means), np.mean(clip_hi), np.mean(clip_lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--cover-test", action="store_true",
                    help="asks you to cover the lens: a locked exposure must NOT brighten the dark")
    a = ap.parse_args()
    cfg = load_config(a.config, {"camera": {"index": a.camera}} if a.camera is not None else None)
    cam = Camera(cfg.camera).open()
    rep = cam.report
    print(f"\nBackend: {rep.backend}")
    stamps, means, hi, lo = collect(cam, a.seconds)
    st = fps_stats(stamps)
    print("\n--- Frame rate (measured from timestamps) ---")
    print(f"requested {cfg.camera.fps} fps, delivered {st['fps']:.1f} fps, jitter {st['interval_std_ms']:.1f} ms, "
          f"dropped {st['dropped_pct']:.1f} %  ({st['frames']} frames)")
    print("\n--- Exposure / gain lock (what the driver reports back) ---")
    for k in rep.readback:
        print(f"{k:14s} asked={rep.requested[k]}  read-back={rep.readback[k]}  obeyed={rep.obeyed[k]}")
    drift = float(means.max() - means.min()) if len(means) else 0.0
    print(f"frame brightness wandered by {drift:.1f} gray levels over {a.seconds:.0f} s (static scene)")

    ae_creeps = None
    if a.cover_test:
        input("\nPress Enter, then COVER THE LENS with your hand for ~6 seconds...")
        _, dark, _, _ = collect(cam, 6.0)
        third = max(1, len(dark) // 3)
        ae_creeps = float(dark[-third:].mean() - dark[:third].mean())
        print(f"brightness while covered rose by {ae_creeps:.1f} levels "
              f"({'auto-exposure is still active' if ae_creeps > 6 else 'consistent with a locked exposure'})")
    cam.close()

    print("\n--- Clipping ---")
    print(f"{hi * 100:.2f} % of pixels near white (>=250), {lo * 100:.2f} % near black (<=5)")

    fps = st["fps"]
    bmax = max_bit_rate(fps, cfg.camera.min_samples_per_symbol)
    locked = rep.exposure_locked
    problems, notes = [], []
    if fps < 20:
        problems.append(f"only {fps:.0f} fps: too slow for a 2 bit/s link with margin")
    if st["dropped_pct"] > 5:
        problems.append(f"{st['dropped_pct']:.0f} % of frames are dropped")
    if hi > 0.05:
        problems.append(f"{hi * 100:.0f} % of the picture is blown out; reduce exposure or aim elsewhere")
    if locked is False or (ae_creeps is not None and ae_creeps > 6):
        notes.append("exposure did NOT stay locked: the picture brightness will pump while auto-exposure "
                     "hunts. The receiver's adaptive threshold copes with slow drift, but expect worse range. "
                     "Try the phone-as-webcam app's manual mode, or a different USB webcam.")
    elif locked is None:
        notes.append("exposure lock is unconfirmed: re-run with --cover-test to check it properly")
    if st["interval_std_ms"] > 12:
        notes.append("frame timing is uneven; the decoder uses timestamps so it copes, but it costs margin")

    print("\n=== VERDICT ===")
    if problems:
        print("NOT SUITABLE as it stands:")
        for p in problems: print("  -", p)
    elif notes:
        print("USABLE, with caveats.")
    else:
        print("SUITABLE.")
    for n in notes: print("  -", n)
    if not problems:
        print(f"Recommended bit rate: {max(2, int(bmax))} bit/s or lower (hard ceiling ~{bmax:.1f}). "
              f"At {fps:.0f} fps each half-bit gets about {fps / (2 * max(2, int(bmax))):.1f} camera samples.")
        print(f"Run the receiver with:  python run.py --bit-rate {max(2, int(bmax))}")


if __name__ == "__main__":
    main()
