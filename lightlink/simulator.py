"""Synthetic camera: a blinking Gaussian spot with the nasty things real cameras do.

Models: sensor noise (brighter pixels noisier), slow global brightness drift, frame-time
jitter, reported-timestamp noise, dropped frames, exposure integration (transitions are
blurred over a frame), transmitter clock error, a static lamp distractor, optional spot
wander, and intervals where the light is blocked.
"""
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from . import protocol


@dataclass
class Frame:
    t: float                 # reported timestamp (what the receiver sees)
    image: np.ndarray        # gray uint8 (H, W)
    index: int
    true_xy: tuple[float, float]


class TxLight:
    """Ground-truth light level (0..1) as a function of true time."""

    def __init__(self, payload: bytes, bit_rate: float, gap_s: float, start_s: float,
                 clock_ppm: float = 0.0, blocks=()):
        self.payload = payload
        self.start = start_s
        self.Ts = 1.0 / (2.0 * bit_rate) * (1.0 + clock_ppm * 1e-6)
        syms = protocol.frame_symbols(payload)
        self.n_frame_syms = len(syms)
        self.symbols = np.array(syms + [0] * int(round(gap_s / self.Ts)), dtype=np.float32)
        self.blocks = [tuple(b) for b in blocks]

    @property
    def frame_period_s(self) -> float:
        return len(self.symbols) * self.Ts

    def tx_frames(self, duration_s: float):
        """[(start, end)] of every transmission that finishes before duration_s."""
        out, k = [], 0
        while True:
            s = self.start + k * self.frame_period_s
            e = s + self.n_frame_syms * self.Ts
            if e > duration_s:
                return out
            out.append((s, e))
            k += 1

    def is_blocked(self, s: float, e: float) -> bool:
        return any(s < b1 and e > b0 for b0, b1 in self.blocks)

    def level(self, t: np.ndarray) -> np.ndarray:
        idx = np.floor((t - self.start) / self.Ts).astype(np.int64)
        on = self.symbols[np.mod(idx, len(self.symbols))]
        on = np.where(t >= self.start, on, 0.0)
        for b0, b1 in self.blocks:
            on = np.where((t >= b0) & (t < b1), 0.0, on)
        return on

    def level_exposed(self, t: float, exposure_s: float, n: int = 8) -> float:
        """Average over the exposure window: why edges look soft in a real camera."""
        return float(self.level(t - exposure_s * np.linspace(0, 1, n)).mean())


def _gauss_patch(cx, cy, sigma, shape):
    h, w = shape
    r = int(np.ceil(4 * sigma))
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 2)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 2)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return (y0, y1, x0, x1), g.astype(np.float32)


class Simulator:
    def __init__(self, sim_cfg, bit_rate: float, gap_s: float, payload: bytes | None = None):
        c = sim_cfg
        self.c = c
        self.W, self.H, self.fps = c.width, c.height, c.fps
        payload = payload if payload is not None else c.payload.encode()
        self.tx = TxLight(payload, bit_rate, gap_s, c.start_s, c.tx_clock_ppm, c.blocks)
        self.rng = np.random.default_rng(c.seed)
        self.duration = c.duration_s
        lamp = c.lamp
        base = np.full((self.H, self.W), c.background, np.float32)
        base += np.linspace(0, 12, self.W, dtype=np.float32)[None, :]  # mild vignette-like ramp
        if lamp.enabled:
            (y0, y1, x0, x1), g = _gauss_patch(lamp.xy[0] * self.W, lamp.xy[1] * self.H,
                                               lamp.sigma_px, (self.H, self.W))
            base[y0:y1, x0:x1] += lamp.amplitude * g
        self.base = base

    def spot_xy(self, t: float) -> tuple[float, float]:
        c = self.c
        x, y = c.spot_xy[0] * self.W, c.spot_xy[1] * self.H
        if c.wander_px:
            x += c.wander_px * np.sin(2 * np.pi * t / 23.0)
            y += 0.6 * c.wander_px * np.sin(2 * np.pi * t / 31.0)
        return x, y

    def frames(self) -> Iterator[Frame]:
        c, rng = self.c, self.rng
        dt = 1.0 / self.fps
        for k in range(int(self.duration * self.fps)):
            if k > 0 and rng.random() < c.drop_prob:
                continue                                  # a dropped frame: simply never arrives
            t_true = k * dt + float(np.clip(rng.normal(0, c.jitter_ms * 1e-3), -0.4 * dt, 0.4 * dt))
            lvl = self.tx.level_exposed(t_true, c.exposure_frac * dt)
            sx, sy = self.spot_xy(t_true)
            img = self.base.copy()
            (y0, y1, x0, x1), g = _gauss_patch(sx, sy, c.spot_sigma_px, (self.H, self.W))
            img[y0:y1, x0:x1] += c.spot_amplitude * lvl * g
            img *= 1.0 + c.drift_amp * np.sin(2 * np.pi * t_true / c.drift_period_s)
            sigma = c.noise_sigma * (1.0 + c.shot_noise_k * np.sqrt(np.clip(img, 0, 255) / 255.0))
            img += sigma * rng.standard_normal(img.shape, dtype=np.float32)
            t_rep = t_true + float(rng.normal(0, c.ts_noise_ms * 1e-3))
            yield Frame(t_rep, np.clip(img, 0, 255).astype(np.uint8), k, (sx, sy))
