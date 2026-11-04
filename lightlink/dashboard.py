"""OpenCV dashboard: pure render function plus logging / recording helpers."""
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

COLORS = {"LOCKED": (80, 200, 80), "DEGRADED": (0, 190, 240), "LOST": (60, 60, 230),
          "SEARCHING": (200, 200, 200)}
PANEL_W = 380


class DashboardState:
    """History the render function needs; kept outside the receiver."""

    def __init__(self, cfg):
        self.c = cfg
        self.trace = deque()           # (t, brightness, threshold)
        self.log = deque(maxlen=cfg.log_lines)
        self.counts = {"valid": 0, "failed": 0}

    def add(self, u):
        if u.brightness is not None:
            self.trace.append((u.t, u.brightness, u.threshold))
        while self.trace and self.trace[0][0] < u.t - self.c.trace_seconds:
            self.trace.popleft()
        for ev in u.events:
            if ev.valid:
                self.counts["valid"] += 1
                text = ev.payload.decode("utf-8", "replace")
                self.log.append((f"{ev.t_start:7.1f}s  OK   {text}", COLORS["LOCKED"]))
            else:
                self.counts["failed"] += 1
                self.log.append((f"{ev.t_start:7.1f}s  FAIL {ev.reason}", COLORS["LOST"]))


def _text(img, s, org, color=(235, 235, 235), scale=0.5, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(frame, u, ds, bit_rate, logging_on=False, recording=False):
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    h, w = img.shape[:2]
    col = COLORS[u.state]
    if u.spot is not None:
        x, y = int(u.spot[0]), int(u.spot[1])
        cv2.rectangle(img, (x - 14, y - 14), (x + 14, y + 14), col, 2)
    scale = h / 720.0 if h > 360 else 1.0
    panel = np.full((max(h, 420), PANEL_W, 3), 28, np.uint8)
    pad = np.zeros((panel.shape[0] - h, w, 3), np.uint8)
    video = np.vstack([img, pad]) if pad.shape[0] > 0 else img

    _text(panel, u.state, (12, 34), col, 1.0, 2)
    _text(panel, f"link quality {u.quality:5.1f} / 100", (12, 62))
    cv2.rectangle(panel, (12, 70), (12 + int(3.5 * u.quality), 82), col, -1)
    cv2.rectangle(panel, (12, 70), (362, 82), (120, 120, 120), 1)
    _text(panel, f"FPS {u.fps:5.1f}   bit rate {bit_rate:g} bps   dropped {u.dropped_total}", (12, 106))
    off = "n/a" if u.offset is None else f"dx {u.offset[0]:+.3f}  dy {u.offset[1]:+.3f}"
    _text(panel, f"offset {off}", (12, 128))
    p = u.parts
    _text(panel, "contrast %.2f  track %.2f  crc %.2f  sat %.2f  drop %.2f" % (
        p["contrast"], p["tracking"], p["crc"], p["saturation"], p["drops"]), (12, 150), scale=0.38)
    flags = ("LOG " if logging_on else "") + ("REC" if recording else "")
    _text(panel, f"valid {ds.counts['valid']}  failed {ds.counts['failed']}   {flags}", (12, 172),
          (80, 200, 240))

    # intensity trace with threshold line
    x0, y0, tw, th = 12, 190, 356, 100
    cv2.rectangle(panel, (x0, y0), (x0 + tw, y0 + th), (90, 90, 90), 1)
    if len(ds.trace) > 2:
        ts = np.array([a[0] for a in ds.trace])
        vs = np.array([a[1] for a in ds.trace])
        lo, hi = vs.min(), max(vs.max(), vs.min() + 10)
        px = x0 + (ts - (u.t - ds.c.trace_seconds)) / ds.c.trace_seconds * tw
        py = y0 + th - (vs - lo) / (hi - lo) * (th - 4) - 2
        cv2.polylines(panel, [np.stack([px, py], 1).astype(np.int32)], False, (255, 200, 80), 1, cv2.LINE_AA)
        thr = ds.trace[-1][2]
        if thr is not None:
            ty = int(y0 + th - (thr - lo) / (hi - lo) * (th - 4) - 2)
            if y0 <= ty <= y0 + th:
                cv2.line(panel, (x0, ty), (x0 + tw, ty), (90, 90, 230), 1)
    _text(panel, "intensity (threshold in red)", (x0, y0 + th + 14), (160, 160, 160), 0.38)

    y = 330
    _text(panel, "decoded messages", (12, y), (160, 160, 160), 0.45)
    for line, c in list(ds.log)[-ds.c.log_lines:]:
        y += 20
        _text(panel, line[:44], (12, y), c, 0.45)
    _text(panel, "keys: l log  v video  q quit", (12, panel.shape[0] - 10), (140, 140, 140), 0.4)
    return np.hstack([video, panel])


class SessionLogger:
    """CSV of per-frame state plus one line per decoded frame. Toggled with 'l'."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.f = self.w = None

    @property
    def on(self):
        return self.f is not None

    def toggle(self):
        if self.on:
            self.f.close()
            self.f = self.w = None
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / time.strftime("session_%Y%m%d_%H%M%S.csv")
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "state", "quality", "dx", "dy", "brightness", "event", "text_or_reason"])
        return path

    def write(self, u):
        if not self.on:
            return
        dx, dy = u.offset if u.offset else ("", "")
        self.w.writerow([f"{u.t:.4f}", u.state, f"{u.quality:.1f}", dx, dy, u.brightness, "", ""])
        for ev in u.events:
            self.w.writerow([f"{ev.t_start:.4f}", u.state, f"{u.quality:.1f}", "", "", "",
                             "VALID" if ev.valid else "FAILED",
                             ev.payload.decode("utf-8", "replace") if ev.valid else ev.reason])


class VideoRecorder:
    """Saves the RAW camera stream plus timestamps so the clip can be replayed through the
    receiver later (scripts/run_replay.py --video)."""

    def __init__(self, directory, fourcc, fps):
        self.dir, self.fourcc, self.fps = Path(directory), fourcc, fps
        self.w = None
        self.stamps = []
        self.path = None

    @property
    def on(self):
        return self.w is not None

    def toggle(self, frame_shape):
        if self.on:
            self.w.release()
            self.path.with_suffix(".avi.ts.csv").write_text("\n".join(f"{t:.6f}" for t in self.stamps))
            self.w, self.stamps = None, []
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / time.strftime("capture_%Y%m%d_%H%M%S.avi")
        h, w = frame_shape[:2]
        self.w = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*self.fourcc),
                                 max(self.fps, 1.0), (w, h))
        return self.path

    def write(self, t, frame):
        if self.on:
            self.w.write(frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            self.stamps.append(t)
