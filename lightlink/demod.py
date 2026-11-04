"""Brightness series -> verified frames.

Pipeline: adaptive threshold -> preamble+sync template search over (clock rate, phase)
-> per-bit decode with a small timing loop locked to the mid-bit transitions -> CRC.
Samples are used at their real timestamps, so jitter and dropped frames do not shift
the bit grid. Only CRC-verified frames carry a payload; everything else is "failed".
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import protocol as P


@dataclass
class FrameEvent:
    t_start: float
    t_end: float
    valid: bool
    reason: str                      # "ok", "bad_crc", "bad_length"
    payload: bytes | None = None     # set only when valid
    raw_bits: list = field(default_factory=list)   # decoded bits after sync (for BER stats)
    corr: float = 0.0


class _BitReader:
    """Reads Manchester bits starting at t0, tracking timing from the mid-bit transitions."""

    def __init__(self, t, x, t0, Tsr, cfg):
        self.t, self.x, self.t0, self.Ts, self.c = t, x, t0, Tsr, cfg
        self.n, self.phase = 0, 0.0

    def _half(self, a, b):
        f = self.c.window_frac
        pts = a + (b - a) * (0.5 + f * (np.arange(5) / 4 - 0.5))
        return float(np.interp(pts, self.t, self.x).mean())

    def read(self, count):
        bits = []
        Ts = self.Ts
        for _ in range(count):
            tb = self.t0 + self.n * 2 * Ts + self.phase
            if tb < self.t[0] or tb + 2 * Ts > self.t[-1]:
                return None                                  # not enough data yet
            d = self._half(tb, tb + Ts) - self._half(tb + Ts, tb + 2 * Ts)
            b = 1 if d > 0 else 0
            # at the true mid-bit transition the signal crosses zero; a leftover value tells
            # us which way our clock is off (late -> value has the second-half sign)
            mid = float(np.interp(tb + Ts, self.t, self.x))
            err = float(np.clip((1 if b else -1) * mid, -1, 1))
            self.phase += self.c.pll_gain * err * self.c.pll_step * Ts
            bits.append(b)
            self.n += 1
        return bits


class Demodulator:
    def __init__(self, cfg, bit_rate: float):
        self.c = cfg
        self.bit_rate = bit_rate
        self.Ts = 1.0 / (2.0 * bit_rate)
        max_bits = P.PREAMBLE_BITS + P.SYNC_BITS + P.LENGTH_BITS + 8 * P.MAX_PAYLOAD + P.CRC_BITS
        self.keep_s = cfg.buffer_margin * max_bits / bit_rate
        self.template = np.array(P.preamble_sync_symbols(), np.float32) * 2 - 1
        self.sync_bits = P.preamble_sync_bits()
        self.reset()

    def reset(self):
        self.t, self.v = deque(), deque()
        self.handled: list[float] = []
        self.contrast = 0.0
        self.threshold = None
        self._last_poll = -1e9

    def push(self, t: float, v: float):
        self.t.append(t)
        self.v.append(v)

    def trace(self):
        return np.fromiter(self.t, float), np.fromiter(self.v, float)

    def _symbol_samples(self, t, x, t0, Tsr):
        k = int((t[-1] - t0) / Tsr) - 1
        if k <= 0:
            return None
        centers = t0 + (np.arange(k) + 0.5) * Tsr
        ok = (centers - 0.25 * Tsr >= t[0])
        s = sum(np.interp(centers + o * Tsr, t, x) for o in (-0.2, 0.0, 0.2)) / 3.0
        return np.where(ok, s, 0.0)

    def _candidates(self, t, x):
        c, n = self.c, len(self.template)
        tnorm = float(np.linalg.norm(self.template))
        found = []
        for r in np.linspace(1 - c.rate_tolerance, 1 + c.rate_tolerance, c.rate_steps):
            Tsr = self.Ts * r
            for p in range(c.phase_steps):
                t0 = t[0] + p * Tsr / c.phase_steps
                s = self._symbol_samples(t, x, t0, Tsr)
                if s is None or len(s) < n:
                    continue
                num = np.correlate(s, self.template, "valid")
                cs = np.concatenate([[0.0], np.cumsum(s * s)])
                energy = np.sqrt(np.maximum(cs[n:] - cs[:-n], 1e-9))
                corr = num / (energy * tnorm)
                for i in np.flatnonzero(corr >= c.corr_threshold):
                    found.append((float(corr[i]), t0 + i * Tsr, Tsr))
        found.sort(reverse=True)
        kept = []
        for corr, ts, Tsr in found:                       # non-maximum suppression in time
            if all(abs(ts - k[1]) > 8 * self.Ts for k in kept):
                kept.append((corr, ts, Tsr))
        return sorted(kept, key=lambda k: k[1])

    def _decode(self, t, x, corr, ts, Tsr):
        """Returns a FrameEvent, 'pending' (need more data) or None (not a real frame)."""
        c = self.c
        rd = _BitReader(t, x, ts, Tsr, c)
        head = rd.read(P.PREAMBLE_BITS + P.SYNC_BITS)
        if head is None:
            return "pending"
        pre_err = sum(1 for b in head[:P.PREAMBLE_BITS] if b != 1)
        sync_err = sum(1 for a, b in zip(head[P.PREAMBLE_BITS:], self.sync_bits[P.PREAMBLE_BITS:]) if a != b)
        if pre_err > c.preamble_max_bit_errors or sync_err > c.sync_max_bit_errors:
            return None
        Tb = 2 * Tsr
        length_bits = rd.read(P.LENGTH_BITS)
        if length_bits is None:
            return "pending"
        length = P.bits_to_bytes(length_bits)[0]
        t_start = ts
        if length > P.MAX_PAYLOAD:
            return FrameEvent(t_start, ts + rd.n * Tb, False, "bad_length", None, length_bits, corr)
        rest = rd.read(8 * length + P.CRC_BITS)
        if rest is None:
            return "pending"
        bits = length_bits + rest
        res = P.parse_body_bits(bits)
        return FrameEvent(t_start, ts + rd.n * Tb, res.ok, res.reason, res.payload, bits, corr)

    def poll(self, now: float) -> list[FrameEvent]:
        c = self.c
        if now - self._last_poll < c.poll_interval_s:
            return []
        self._last_poll = now
        while self.t and self.t[0] < now - self.keep_s:
            self.t.popleft()
            self.v.popleft()
        self.handled = [h for h in self.handled if h > now - self.keep_s]
        if len(self.t) < 20:
            return []
        t, v = self.trace()
        lo, hi = np.percentile(v, [c.low_pct, c.high_pct])
        self.contrast, self.threshold = float(hi - lo), float((hi + lo) / 2)
        if self.contrast < c.min_contrast:
            return []
        x = np.clip((v - self.threshold) / self.contrast * 2.0, -1.5, 1.5)
        events = []
        for corr, ts, Tsr in self._candidates(t, x):
            if any(abs(ts - h) < c.dup_tol_bits * 2 * self.Ts for h in self.handled):
                continue
            ev = self._decode(t, x, corr, ts, Tsr)
            if ev is None or ev == "pending":
                continue
            events.append(ev)
            self.handled.append(ev.t_start)
        return events
