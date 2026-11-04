"""Webcam capture with real timestamps, true-FPS measurement and honest exposure-lock reporting."""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

_BACKENDS = {"dshow": getattr(cv2, "CAP_DSHOW", 0), "msmf": getattr(cv2, "CAP_MSMF", 0),
             "any": getattr(cv2, "CAP_ANY", 0)}


@dataclass
class LockReport:
    backend: str = ""
    requested: dict = field(default_factory=dict)
    readback: dict = field(default_factory=dict)
    obeyed: dict = field(default_factory=dict)   # name -> True / False / None (unknown)

    @property
    def exposure_locked(self):
        """True only if the driver both accepted manual mode and the requested exposure."""
        vals = [self.obeyed.get("auto_exposure"), self.obeyed.get("exposure")]
        if any(v is False for v in vals):
            return False
        return True if all(v is True for v in vals) else None


class Camera:
    def __init__(self, cfg):
        self.c = cfg
        self.cap = None
        self.report = LockReport()

    def open(self):
        for name in self.c.backends:
            cap = cv2.VideoCapture(self.c.index, _BACKENDS[name])
            if cap.isOpened():
                self.cap, self.report.backend = cap, name
                break
            cap.release()
        if self.cap is None:
            raise RuntimeError("no camera could be opened (tried: %s)" % ", ".join(self.c.backends))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.c.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.c.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.c.fps)
        if self.c.lock_exposure:
            self.lock_exposure()
        return self

    def lock_exposure(self):
        """Ask for manual exposure/gain, then READ BACK what the driver says. Drivers often
        accept a set() call silently, so a read-back is the only evidence we can log."""
        cap, rep, tol = self.cap, self.report, self.c.readback_tol
        # AUTO_EXPOSURE means different things per backend: 0.25 = manual on DirectShow/V4L2,
        # 1 on some MSMF builds. We try both and keep whichever the driver reports back.
        for mode in (0.25, 1.0, 0.0):
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, mode)
            if abs(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) - mode) <= tol / 2:
                break
        wanted = {"auto_exposure": None, "exposure": self.c.exposure, "gain": self.c.gain}
        cap.set(cv2.CAP_PROP_EXPOSURE, self.c.exposure)
        cap.set(cv2.CAP_PROP_GAIN, self.c.gain)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        for _ in range(self.c.settle_frames):
            cap.grab()
        props = {"auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE, "exposure": cv2.CAP_PROP_EXPOSURE,
                 "gain": cv2.CAP_PROP_GAIN}
        for k, p in props.items():
            rep.readback[k] = cap.get(p)
            rep.requested[k] = wanted[k]
            if k == "auto_exposure":
                # manual shows up as 0.25 (DirectShow/V4L2), 1 (some MSMF), or 0
                rep.obeyed[k] = rep.readback[k] in (0.25, 1.0, 0.0)
            else:
                rep.obeyed[k] = abs(rep.readback[k] - wanted[k]) <= tol
        return rep

    def read(self):
        """(timestamp, frame). Timestamp is taken right after grab(), the closest we get to
        the moment the sensor finished exposing, on a monotonic clock."""
        if not self.cap.grab():
            return None
        t = time.perf_counter()
        ok, frame = self.cap.retrieve()
        return (t, frame) if ok else None

    def frames(self):
        while True:
            r = self.read()
            if r is None:
                return
            yield r

    def close(self):
        if self.cap:
            self.cap.release()


def fps_stats(stamps, drop_factor=1.8) -> dict:
    """True frame rate from timestamps (what the camera delivers, not what it claims)."""
    dt = np.diff(np.asarray(stamps, float))
    if len(dt) < 3:
        return {"fps": 0.0, "interval_std_ms": 0.0, "dropped_pct": 0.0, "frames": len(stamps)}
    med = float(np.median(dt))
    dropped = int(sum(round(d / med) - 1 for d in dt if d > drop_factor * med))
    return {"fps": 1.0 / med, "mean_fps": 1.0 / float(dt.mean()),
            "interval_std_ms": float(dt.std() * 1000), "dropped_pct": 100.0 * dropped / (len(dt) + dropped),
            "frames": len(stamps)}


def max_bit_rate(fps: float, min_samples_per_symbol: float, ceiling: float = 15.0) -> float:
    """Manchester sends two symbols per bit; each symbol needs a few camera samples."""
    return min(ceiling, fps / (2.0 * min_samples_per_symbol))
