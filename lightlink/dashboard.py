"""OpenCV dashboard: a pure render function that adapts to any window size, plus logging/recording helpers.

render() draws at the real window size (no stretching): wide windows get a side panel, tall/narrow ones a
panel underneath, and fonts, bars and spacing scale with the window. Sections that do not fit are dropped
from the bottom up, so a tiny window still shows the state, quality and message log first.
"""
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# BGR colours
BG, CARD, LINE = (26, 23, 21), (42, 38, 35), (74, 69, 65)
TEXT, MUTED, DARK = (238, 234, 230), (150, 144, 138), (20, 18, 16)
STATE_COLOR = {"LOCKED": (110, 205, 90), "DEGRADED": (0, 178, 255), "LOST": (80, 80, 240),
               "SEARCHING": (200, 195, 190)}
OK, BAD, TRACE = (110, 205, 90), (80, 80, 240), (255, 200, 90)
FONT = cv2.FONT_HERSHEY_SIMPLEX
DESIGN_W, DESIGN_H = 1280, 720           # size at which scale == 1
DEFAULT_SIZE = (DESIGN_W, DESIGN_H)


class DashboardState:
    """History the render function needs; kept outside the receiver."""

    def __init__(self, cfg):
        self.c = cfg
        self.trace = deque()           # (t, brightness, threshold)
        self.log = deque(maxlen=60)    # (time, ok, text); render shows as many as fit
        self.counts = {"valid": 0, "failed": 0}
        self.buttons = {}              # name -> (x0, y0, x1, y1) of clickable areas, set by render()

    def add(self, u):
        if u.brightness is not None:
            self.trace.append((u.t, u.brightness, u.threshold))
        while self.trace and self.trace[0][0] < u.t - self.c.trace_seconds:
            self.trace.popleft()
        for ev in u.events:
            if ev.valid:
                self.counts["valid"] += 1
                self.log.append((ev.t_start, True, ev.payload.decode("utf-8", "replace")))
            else:
                self.counts["failed"] += 1
                self.log.append((ev.t_start, False, "failed: " + ev.reason))


# --- drawing helpers ---------------------------------------------------------------------------
def _rrect(img, x0, y0, x1, y1, r, color):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    r = int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    if r < 2:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
        return
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _card(img, x0, y0, x1, y1, s):
    _rrect(img, x0, y0, x1, y1, 8 * s, LINE)
    _rrect(img, x0 + 1, y0 + 1, x1 - 1, y1 - 1, 8 * s, CARD)


def _text_w(text, scale, thick=1):
    return cv2.getTextSize(text, FONT, scale, thick)[0][0]


def _put(img, text, x, y, scale, color=TEXT, thick=1, max_w=None, right=False):
    """y is the text baseline. Truncates with '...' to max_w; right=True right-aligns at x."""
    if max_w is not None and _text_w(text, scale, thick) > max_w:
        while len(text) > 1 and _text_w(text + "...", scale, thick) > max_w:
            text = text[:-1]
        text += "..."
    if right:
        x = x - _text_w(text, scale, thick)
    cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def _bar(img, x0, y0, w, h, frac, color, s):
    _rrect(img, x0, y0, x0 + w, y0 + h, h / 2, (58, 54, 50))
    fw = int(w * max(0.0, min(1.0, frac)))
    if fw > h:
        _rrect(img, x0, y0, x0 + fw, y0 + h, h / 2, color)
    elif fw > 1:
        _rrect(img, x0, y0, x0 + h, y0 + h, h / 2, color)


def _letterbox(frame, w, h):
    """Fit frame inside (w, h) keeping aspect. Returns canvas, scale, offset_x, offset_y."""
    fh, fw = frame.shape[:2]
    k = min(w / fw, h / fh)
    nw, nh = max(1, int(fw * k)), max(1, int(fh * k))
    img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    return img, k, (w - nw) // 2, (h - nh) // 2


# --- panels --------------------------------------------------------------------------------------
def _draw_video(canvas, frame, u, rect, s, recording):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    m = int(10 * s)
    aw, ah = max(1, w - 2 * m), max(1, h - 2 * m)
    img, k, ox, oy = _letterbox(frame, aw, ah)
    px, py = x0 + m + ox, y0 + m + oy
    canvas[py:py + img.shape[0], px:px + img.shape[1]] = img
    col = STATE_COLOR[u.state]
    if u.spot is not None:                      # corner-bracket box on the tracked spot
        cx, cy = int(px + u.spot[0] * k), int(py + u.spot[1] * k)
        r, L = int(18 * s), int(8 * s)
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            ax, ay = cx + sx * r, cy + sy * r
            cv2.line(canvas, (ax, ay), (ax - sx * L, ay), col, max(1, int(2 * s)), cv2.LINE_AA)
            cv2.line(canvas, (ax, ay), (ax, ay - sy * L), col, max(1, int(2 * s)), cv2.LINE_AA)
    if recording:
        rx = px + img.shape[1] - int(14 * s)
        cv2.circle(canvas, (rx - int(46 * s), py + int(18 * s)), int(6 * s) + 1, BAD, -1, cv2.LINE_AA)
        _put(canvas, "REC", rx, py + int(24 * s), 0.55 * s, TEXT, 1, right=True)


def _draw_panel(canvas, u, ds, bit_rate, rect, s, logging_on, recording):
    x0, y0, x1, y1 = rect
    pw, ph = x1 - x0, y1 - y0
    pad = int(12 * s)
    gap = int(10 * s)
    ix0, ix1 = x0 + pad, x1 - pad
    iw = ix1 - ix0
    y = y0 + pad
    bottom = y1 - pad
    col = STATE_COLOR[u.state]
    foot_h = int(22 * s)
    bottom -= foot_h

    # 1. state + quality ---------------------------------------------------------------------
    hh = int(54 * s)
    if bottom - y >= hh:
        _card(canvas, ix0, y, ix1, y + hh, s)
        _rrect(canvas, ix0 + 8 * s, y + 8 * s, ix0 + 8 * s + _text_w(u.state, 0.8 * s, 2) + 22 * s,
               y + hh - 8 * s, 7 * s, col)
        _put(canvas, u.state, ix0 + 19 * s, y + hh / 2 + 9 * s, 0.8 * s, DARK, 2)
        _put(canvas, "link quality", ix1 - 14 * s, y + 18 * s, 0.38 * s, MUTED, 1, right=True)
        _put(canvas, f"{u.quality:.0f}", ix1 - 14 * s, y + hh - 9 * s, 0.95 * s, col, 2, right=True)
        y += hh + gap
        if bottom - y >= 14 * s:
            _bar(canvas, ix0, y, iw, int(10 * s), u.quality / 100.0, col, s)
            y += int(10 * s) + gap

    # 1b. bit rate control (must equal the transmitter's) -------------------------------------
    ds.buttons = {}
    rh = int(46 * s)
    if bottom - y >= rh + 70 * s:
        _card(canvas, ix0, y, ix1, y + rh, s)
        _put(canvas, "BIT RATE", ix0 + 12 * s, y + 19 * s, 0.38 * s, MUTED)
        _put(canvas, "match the sender", ix0 + 12 * s, y + 36 * s, 0.32 * s, MUTED, max_w=iw * 0.4)
        bw, by0, by1 = int(36 * s), y + int(6 * s), y + rh - int(6 * s)
        plus_x = ix1 - int(10 * s) - bw
        minus_x = plus_x - int(86 * s) - bw
        for name, bx, glyph in (("rate_down", minus_x, "-"), ("rate_up", plus_x, "+")):
            _rrect(canvas, bx, by0, bx + bw, by1, 9 * s, (70, 66, 62))
            gw = _text_w(glyph, 0.9 * s, 2)
            _put(canvas, glyph, bx + (bw - gw) / 2, (by0 + by1) / 2 + 7 * s, 0.9 * s, TEXT, 2)
            ds.buttons[name] = (bx, by0, bx + bw, by1)
        label = f"{bit_rate:g} bps"
        lx = (minus_x + bw + plus_x) / 2 - _text_w(label, 0.62 * s, 2) / 2
        _put(canvas, label, lx, (by0 + by1) / 2 + 7 * s, 0.62 * s, TEXT, 2)
        y += rh + gap

    # 2. stats grid ---------------------------------------------------------------------------
    off = "--" if u.offset is None else f"{u.offset[0]:+.2f} {u.offset[1]:+.2f}"
    stats = [("FPS", f"{u.fps:.1f}"), ("DROPPED", str(u.dropped_total)), ("OFFSET dx dy", off),
             ("VALID", str(ds.counts["valid"])), ("FAILED", str(ds.counts["failed"]))]
    cols = 3 if iw >= 300 * s else 2
    rows = -(-len(stats) // cols)
    cell_h = int(44 * s)
    if bottom - y >= cell_h * rows * 0.6:
        rows = min(rows, max(1, int((bottom - y) // cell_h)))
        _card(canvas, ix0, y, ix1, y + cell_h * rows, s)
        cw = iw / cols
        for i, (lab, val) in enumerate(stats[:rows * cols]):
            cx, cy = ix0 + (i % cols) * cw + 12 * s, y + (i // cols) * cell_h
            vcol = TEXT
            if lab == "FAILED" and ds.counts["failed"]:
                vcol = BAD
            if lab == "VALID" and ds.counts["valid"]:
                vcol = OK
            _put(canvas, lab, cx, cy + 16 * s, 0.38 * s, MUTED, 1, max_w=cw - 14 * s)
            _put(canvas, val, cx, cy + 36 * s, 0.6 * s, vcol, 1, max_w=cw - 14 * s)
        y += cell_h * rows + gap

    # 3. quality components ---------------------------------------------------------------------
    names = [("contrast", "contrast"), ("tracking", "tracking"), ("crc", "crc"), ("saturation", "no clipping"),
             ("drops", "no drops")]
    row_h = int(18 * s)
    comp_h = row_h * len(names) + int(12 * s)
    if bottom - y >= comp_h + 90 * s:
        _card(canvas, ix0, y, ix1, y + comp_h, s)
        for i, (key, lab) in enumerate(names):
            ry = y + int(6 * s) + i * row_h
            v = u.parts.get(key, 0.0)
            _put(canvas, lab, ix0 + 12 * s, ry + 12 * s, 0.38 * s, MUTED)
            bx = ix0 + 12 * s + 92 * s
            _bar(canvas, bx, ry + 4 * s, ix1 - 12 * s - 40 * s - bx, int(7 * s), v, col, s)
            _put(canvas, f"{v:.2f}", ix1 - 12 * s, ry + 12 * s, 0.38 * s, TEXT, right=True)
        y += comp_h + gap

    # 4. brightness trace -----------------------------------------------------------------------
    remaining = bottom - y
    log_min = int(70 * s)
    trace_h = int(min(130 * s, max(0, remaining - log_min - gap)))
    if trace_h >= 50 * s:
        _card(canvas, ix0, y, ix1, y + trace_h, s)
        _put(canvas, "brightness  (red line = threshold)", ix0 + 12 * s, y + 16 * s, 0.38 * s, MUTED,
             max_w=iw - 24 * s)
        gx0, gx1, gy0, gy1 = ix0 + 10 * s, ix1 - 10 * s, y + 24 * s, y + trace_h - 8 * s
        if len(ds.trace) > 2:
            ts = np.array([a[0] for a in ds.trace])
            vs = np.array([a[1] for a in ds.trace])
            lo, hi = vs.min(), max(vs.max(), vs.min() + 10)
            px = gx0 + (ts - (u.t - ds.c.trace_seconds)) / ds.c.trace_seconds * (gx1 - gx0)
            py = gy1 - (vs - lo) / (hi - lo) * (gy1 - gy0)
            cv2.polylines(canvas, [np.stack([px, py], 1).astype(np.int32)], False, TRACE,
                          max(1, int(1.5 * s)), cv2.LINE_AA)
            thr = ds.trace[-1][2]
            if thr is not None and lo <= thr <= hi:
                ty = int(gy1 - (thr - lo) / (hi - lo) * (gy1 - gy0))
                for xx in range(int(gx0), int(gx1), max(4, int(8 * s))):
                    cv2.line(canvas, (xx, ty), (xx + max(2, int(4 * s)), ty), BAD, 1)
        y += trace_h + gap

    # 5. message log (all remaining space) -----------------------------------------------------
    if bottom - y >= 44 * s:
        _card(canvas, ix0, y, ix1, bottom, s)
        _put(canvas, "messages", ix0 + 12 * s, y + 16 * s, 0.38 * s, MUTED)
        line_h = int(22 * s)
        n = max(1, int((bottom - y - 26 * s) // line_h))
        shown = list(ds.log)[-n:]
        if not shown:
            _put(canvas, "waiting for a frame...", ix0 + 12 * s, y + 42 * s, 0.45 * s, MUTED)
        for i, (t, ok, text) in enumerate(reversed(shown)):       # newest first
            ly = y + 26 * s + i * line_h + 14 * s
            tag, tcol = ("OK", OK) if ok else ("FAIL", BAD)
            _rrect(canvas, ix0 + 10 * s, ly - 13 * s, ix0 + 10 * s + 38 * s, ly + 4 * s, 4 * s, tcol)
            _put(canvas, tag, ix0 + 14 * s, ly, 0.4 * s, DARK, 1)
            _put(canvas, text, ix0 + 56 * s, ly, 0.5 * s, TEXT if ok else MUTED, 1,
                 max_w=iw - 56 * s - 70 * s)
            _put(canvas, f"{t:.0f}s", ix1 - 10 * s, ly, 0.38 * s, MUTED, 1, right=True)

    # footer ------------------------------------------------------------------------------------
    fy = y1 - pad + int(2 * s)
    hint = "+/- rate   L log   V video   F full   Q quit"
    _put(canvas, hint, ix0, fy, 0.36 * s, MUTED, 1, max_w=iw - 60 * s)
    flags = [("LOG", logging_on), ("REC", recording)]
    fx = ix1
    for name, on in flags:
        if on:
            _put(canvas, name, fx, fy, 0.4 * s, BAD if name == "REC" else OK, 1, right=True)
            fx -= _text_w(name, 0.4 * s) + 10 * s


def render(frame, u, ds, bit_rate, logging_on=False, recording=False, size=None):
    """Draw the whole dashboard into an image of exactly `size` = (width, height)."""
    W, H = size or DEFAULT_SIZE
    W, H = max(int(W), 160), max(int(H), 120)
    s = float(np.clip(min(W / DESIGN_W, H / DESIGN_H), 0.45, 3.0))
    canvas = np.full((H, W, 3), BG, np.uint8)
    if W >= 1.15 * H:                                       # landscape: panel on the right
        pw = int(np.clip(W * 0.32, 300 * s, 520 * s))
        pw = min(pw, W - 120)
        video, panel = (0, 0, W - pw, H), (W - pw, 0, W, H)
    else:                                                   # portrait / narrow: panel underneath
        fh, fw = frame.shape[:2]
        vh = int(min(W * fh / fw + 20 * s, H * 0.5))
        video, panel = (0, 0, W, vh), (0, vh, W, H)
    _draw_video(canvas, frame, u, video, s, recording)
    _draw_panel(canvas, u, ds, bit_rate, panel, s, logging_on, recording)
    return canvas


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
