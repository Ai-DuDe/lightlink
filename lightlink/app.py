"""Shared dashboard loop for live camera and replay."""
import time

import cv2

from .dashboard import DashboardState, SessionLogger, VideoRecorder, render
from .receiver import Receiver


def run_dashboard(cfg, source, realtime=False, headless_frames=None):
    """source yields (t, frame). Keys: l = start/stop CSV log, v = start/stop raw video, q = quit."""
    d = cfg.dashboard
    rx = Receiver(cfg)
    ds = DashboardState(d)
    logger = SessionLogger(d.log_dir)
    rec = VideoRecorder(d.log_dir, d.video_fourcc, cfg.camera.fps)
    wall0 = t0 = None
    n = 0
    try:
        for t, frame in source:
            u = rx.process(t, frame)
            ds.add(u)
            logger.write(u)
            rec.write(t, frame)
            n += 1
            if headless_frames and n >= headless_frames:
                break
            if realtime:
                t0 = t if t0 is None else t0
                wall0 = time.perf_counter() if wall0 is None else wall0
                lag = (t - t0) - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(min(lag, 0.1))
            cv2.imshow(d.window_name, render(frame, u, ds, rx.bit_rate, logger.on, rec.on))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("l"):
                p = logger.toggle()
                print("logging ->", p if p else "stopped")
            if key == ord("v"):
                p = rec.toggle(frame.shape)
                print("recording ->", p if p else "stopped (timestamps saved next to the .avi)")
    finally:
        if logger.on:
            logger.toggle()
        if rec.on:
            rec.toggle(None)
        cv2.destroyAllWindows()
    return ds
