"""Find a blinking source by temporal variation.

We keep a slow-decaying average of squared frame-to-frame differences. A static lamp,
however bright, has ~zero difference energy; a blinking LED/laser lights up this map.
The peak must stand well above the robust noise floor and be compact, so scene-wide
changes (someone walks past, exposure jumps) are not mistaken for the source.
"""
import cv2
import numpy as np


class BlinkDetector:
    def __init__(self, cfg, width: int, height: int):
        self.c = cfg
        self.scale = min(1.0, cfg.analysis_width / width)
        self.size = (max(8, round(width * self.scale)), max(8, round(height * self.scale)))
        w, h = self.size
        self.energy = np.zeros((h, w), np.float32)
        self.blurred = None
        self.prev = None
        self.last_t = None
        self.t0 = None
        self.n = 0
        self.floor = 0.0
        self.mad = 1.0

    def update(self, gray: np.ndarray, t: float):
        small = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA) if self.scale < 1 else gray
        small = small.astype(np.float32)
        if self.t0 is None:
            self.t0 = t
        if self.prev is not None:
            a = min(1.0, max(t - self.last_t, 1e-3) / self.c.energy_tau_s)
            d = small - self.prev
            self.energy += a * (d * d - self.energy)
            self.blurred = cv2.GaussianBlur(self.energy, (0, 0), self.c.blur_sigma_px)
            if self.n % self.c.stats_every == 0:
                self.floor = float(np.median(self.blurred))
                # 1.4826 turns MAD into a standard deviation; the floor keeps it nonzero
                self.mad = 1.4826 * float(np.median(np.abs(self.blurred - self.floor))) + 1e-3
        self.prev, self.last_t = small, t
        self.n += 1

    # coordinate helpers: cv2.INTER_AREA maps pixel centres, so use the exact transform
    def _to_full(self, xs, ys):
        s = self.scale
        return (xs + 0.5) / s - 0.5, (ys + 0.5) / s - 0.5

    def _accept(self, peak: float) -> bool:
        ex = peak - self.floor
        return ex >= self.c.min_excess and ex / self.mad >= self.c.min_z

    def _warm(self, t) -> bool:
        return self.blurred is not None and t - self.t0 >= self.c.warmup_s

    def _refine(self, x, y, peak):
        r = int(np.ceil(3 * self.c.blur_sigma_px))
        h, w = self.blurred.shape
        x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
        half = self.floor + 0.5 * (peak - self.floor)
        wgt = np.clip(self.blurred[y0:y1, x0:x1] - half, 0, None)
        if wgt.sum() > 0:
            ys, xs = np.mgrid[y0:y1, x0:x1]
            x, y = float((wgt * xs).sum() / wgt.sum()), float((wgt * ys).sum() / wgt.sum())
        fx, fy = self._to_full(x, y)
        return float(fx), float(fy), float(peak)

    def find_peak(self, t):
        """Strongest blinking source anywhere in the frame, or None."""
        if not self._warm(t):
            return None
        _, mx, _, (x, y) = cv2.minMaxLoc(self.blurred)
        if not self._accept(mx):
            return None
        half = self.floor + 0.5 * (mx - self.floor)
        if cv2.countNonZero((self.blurred > half).astype(np.uint8)) > \
                self.c.max_blob_fraction * self.blurred.size:
            return None
        return self._refine(x, y, mx)

    def peak_near(self, x, y, radius, t):
        """Strongest blinking source within `radius` full-res pixels of (x, y), or None."""
        if not self._warm(t):
            return None
        h, w = self.blurred.shape
        cx, cy, r = (x + 0.5) * self.scale - 0.5, (y + 0.5) * self.scale - 0.5, radius * self.scale
        x0, x1 = int(max(0, np.floor(cx - r))), int(min(w, np.ceil(cx + r) + 1))
        y0, y1 = int(max(0, np.floor(cy - r))), int(min(h, np.ceil(cy + r) + 1))
        if x1 <= x0 or y1 <= y0:
            return None
        _, mx, _, (px, py) = cv2.minMaxLoc(self.blurred[y0:y1, x0:x1])
        if not self._accept(mx):
            return None
        return self._refine(px + x0, py + y0, mx)
