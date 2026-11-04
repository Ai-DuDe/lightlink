"""Full receive pipeline: frame in -> state, link quality, spot offset and verified messages out."""
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .demod import Demodulator, FrameEvent
from .detector import BlinkDetector
from .link import LinkMonitor, SEARCHING
from .tracker import SpotTracker


@dataclass
class Update:
    t: float
    state: str
    quality: float                       # 0..100
    parts: dict
    spot: tuple | None                   # (x, y) full-res pixels while a track exists
    offset: tuple | None                 # (dx, dy) normalised, see Receiver docstring
    tracking_conf: float
    brightness: float | None
    threshold: float | None
    fps: float
    dropped_total: int
    events: list = field(default_factory=list)   # FrameEvent decoded on this frame


class SpotSampler:
    """Brightness of the tracked spot: mean of the k brightest pixels in a small window.
    Top-k (not a single max) keeps noise down; a small window tolerates sub-pixel drift."""

    def __init__(self, cfg):
        self.c = cfg

    def sample(self, gray, x, y):
        h, w = gray.shape
        r = self.c.roi_half_px
        xi, yi = int(round(x)), int(round(y))
        roi = gray[max(0, yi - r):min(h, yi + r + 1), max(0, xi - r):min(w, xi + r + 1)]
        if roi.size == 0:
            return None, False
        flat = roi.ravel()
        k = min(self.c.top_k, flat.size)
        bright = float(np.partition(flat, -k)[-k:].mean())
        # a saturated *median* means the surroundings are blown out: no contrast to find
        return bright, bool(np.median(flat) >= self.c.sat_level)


class Receiver:
    """Output convention: offset = (dx, dy) where dx = (x - cx) / cx and dy = (y - cy) / cy,
    cx, cy the image centre. +dx = spot right of centre, +dy = spot below centre, range about
    -1..1. It is None unless a track exists, so a consumer (later: pan-tilt) can trust it."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.bit_rate = cfg.protocol.bit_rate_bps
        self.det = None
        self.trk = SpotTracker(cfg.tracker)
        self.sampler = SpotSampler(cfg.sampler)
        self.demod = Demodulator(cfg.demod, self.bit_rate)
        self.link = LinkMonitor(cfg.link)
        self.shape = None
        self.prev_t = None
        self.dts = deque(maxlen=60)
        self.dropped_total = 0
        self.candidate = None            # (x, y, since) waiting for confirmation
        self.last_brightness = None

    def set_bit_rate(self, rate: float):
        """Change the expected bit rate live (must match the transmitter). Clears decoder state
        so samples collected at the old rate cannot produce a frame at the new one."""
        rate = float(rate)
        if not 2.0 <= rate <= 15.0:
            raise ValueError(f"bit rate must be 2..15, got {rate}")
        self.bit_rate = rate
        self.demod = Demodulator(self.cfg.demod, rate)
        self.link.reset_evidence()

    def _acquire(self, t):
        pk = self.det.find_peak(t)
        c = self.cfg.detector
        if pk is None:
            self.candidate = None
            return
        x, y, _ = pk
        if self.candidate and np.hypot(x - self.candidate[0], y - self.candidate[1]) < c.confirm_radius_px:
            if t - self.candidate[2] >= c.confirm_s:
                self.trk.start(x, y, t)
                self.demod.reset()
                self.link.reset_evidence()
                self.candidate = None
        else:
            self.candidate = (x, y, t)

    def _track(self, t):
        trk, c = self.trk, self.cfg.tracker
        trk.predict(t)
        px, py = trk.pos
        meas = self.det.peak_near(px, py, c.gate_px, t)
        rival = None
        if meas is None:
            far = self.det.find_peak(t)
            if far is not None and np.hypot(far[0] - px, far[1] - py) > c.gate_px:
                rival = far
        trk.update(None if meas is None else meas[:2], rejected=rival is not None)
        if rival is not None and trk.silence(t) > c.reacquire_after_s:
            trk.start(rival[0], rival[1], t)           # the old source went quiet; follow the new
            self.demod.reset()
            self.link.reset_evidence()
        elif trk.silence(t) > c.lost_timeout_s:
            trk.drop()
            self.demod.reset()
            self.link.reset_evidence()

    def process(self, t: float, frame: np.ndarray) -> Update:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape
        if self.det is None:
            self.det = BlinkDetector(self.cfg.detector, w, h)
            self.shape = (w, h)
        lc = self.cfg.link
        if self.prev_t is not None:
            dt = t - self.prev_t
            med = float(np.median(self.dts)) if len(self.dts) >= 5 else None
            dropped = int(round(dt / med)) - 1 if med and dt > lc.drop_factor * med else 0
            self.dropped_total += dropped
            self.link.note_frame_interval(t, dropped)
            if dropped == 0:
                self.dts.append(dt)
            elif med is None:
                self.dts.append(dt)
        self.prev_t = t

        self.det.update(gray, t)
        if self.trk.active:
            self._track(t)
        else:
            self._acquire(t)

        events, brightness = [], None
        if self.trk.active:
            x, y = self.trk.pos
            brightness, sat = self.sampler.sample(gray, x, y)
            self.link.note_saturation(sat)
            if brightness is not None:
                self.demod.push(t, brightness)
                events = self.demod.poll(t)
                for ev in events:
                    self.link.note_event(ev)
        conf = self.trk.confidence(t)
        self.link.compute_quality(t, self.demod.contrast, conf)
        state = self.link.update_state(t, self.trk.active, conf)

        spot = self.trk.pos if self.trk.active else None
        offset = None
        if spot is not None:
            cx, cy = (w - 1) / 2, (h - 1) / 2
            offset = ((spot[0] - cx) / cx, (spot[1] - cy) / cy)
        fps = 1.0 / float(np.median(self.dts)) if len(self.dts) >= 5 else 0.0
        return Update(t, state, 100.0 * self.link.quality, dict(self.link.parts), spot, offset,
                      conf, brightness, self.demod.threshold, fps, self.dropped_total, events)
