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
    size, fullscreen = (1280, 720), False
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            for name, (x0, y0, x1, y1) in ds.buttons.items():
                if x0 <= x <= x1 and y0 <= y <= y1:
                    clicks.append(name)

    def step_rate(direction):
        rates = list(d.rates)
        i = min(range(len(rates)), key=lambda k: abs(rates[k] - rx.bit_rate))
        new = rates[max(0, min(len(rates) - 1, i + direction))]
        if new != rx.bit_rate:
            rx.set_bit_rate(new)
            print(f"bit rate -> {new:g} bps (the transmitter must use the same)")

    if headless_frames is None:
        # resizable: the dashboard is redrawn at the real window size, so maximise/resize just work
        cv2.namedWindow(d.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(d.window_name, *size)
        cv2.setMouseCallback(d.window_name, on_mouse)
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
            try:
                _, _, ww, wh = cv2.getWindowImageRect(d.window_name)
                if ww >= 160 and wh >= 120:          # ignore the 0x0 reported while minimised
                    size = (ww, wh)
            except cv2.error:
                pass
            cv2.imshow(d.window_name, render(frame, u, ds, rx.bit_rate, logger.on, rec.on, size))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            while clicks:
                name = clicks.pop(0)
                if name == "dark":
                    ds.dark = not ds.dark
                else:
                    step_rate(1 if name == "rate_up" else -1)
            if key == ord("d"):
                ds.dark = not ds.dark
            if key in (ord("="), ord("+"), ord("]")):
                step_rate(1)
            if key in (ord("-"), ord("_"), ord("[")):
                step_rate(-1)
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(d.window_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
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
