"""Link quality score and the SEARCHING / LOCKED / DEGRADED / LOST state machine.

Rule of thumb: move up slowly, move down immediately. LOCKED needs recent proof
(a CRC-valid frame), not just a steady-looking spot.
"""
from collections import deque

import numpy as np

SEARCHING, LOCKED, DEGRADED, LOST = "SEARCHING", "LOCKED", "DEGRADED", "LOST"


class LinkMonitor:
    def __init__(self, cfg):
        self.c = cfg
        self.state = SEARCHING
        self.attempts = deque(maxlen=cfg.crc_history)   # True = CRC passed
        self.last_valid_t = -1e9
        self.last_valid_dur = 0.0
        self.sat_ema = 0.0
        self.intervals = deque()                        # (t, dropped frames in that interval)
        self._good_since = None
        self._lost_since = 0.0
        self.parts = {k: 0.0 for k in cfg.weights.to_dict()}
        self.quality = 0.0

    # --- inputs -------------------------------------------------------------------------
    def note_frame_interval(self, t, dropped: int):
        self.intervals.append((t, dropped))
        while self.intervals and self.intervals[0][0] < t - self.c.drop_window_s:
            self.intervals.popleft()

    def note_saturation(self, saturated: bool):
        self.sat_ema += 0.02 * ((1.0 if saturated else 0.0) - self.sat_ema)

    def note_event(self, ev):
        self.attempts.append(ev.valid)
        if ev.valid:
            self.last_valid_t = ev.t_end
            self.last_valid_dur = ev.t_end - ev.t_start

    def reset_evidence(self):
        self.attempts.clear()
        self.last_valid_t = -1e9

    # --- derived ------------------------------------------------------------------------
    def valid_window(self) -> float:
        period = self.last_valid_dur + self.c.expected_gap_s
        return self.c.valid_window_factor * period if self.last_valid_dur else 0.0

    def recent_valid(self, t) -> bool:
        return t - self.last_valid_t <= self.valid_window()

    def compute_quality(self, t, contrast, conf) -> float:
        c = self.c
        n = len(self.intervals)
        drops = sum(d for _, d in self.intervals)
        drop_frac = drops / (n + drops) if n else 0.0
        crc_rate = (sum(self.attempts) / len(self.attempts)) if self.attempts else 0.0
        self.parts = {
            "contrast": float(np.clip(contrast / c.contrast_good, 0, 1)),
            "tracking": float(np.clip(conf, 0, 1)),
            # no CRC-valid frame lately counts as zero, however good the old pass rate was
            "crc": crc_rate if self.recent_valid(t) else 0.0,
            "saturation": float(np.clip(1.0 - self.sat_ema * 4, 0, 1)),
            "drops": float(np.clip(1.0 - drop_frac / c.max_drop_fraction, 0, 1)),
        }
        w = c.weights.to_dict()
        self.quality = sum(w[k] * self.parts[k] for k in w) / sum(w.values())
        return self.quality

    # --- state machine ------------------------------------------------------------------
    def update_state(self, t, tracking: bool, conf: float):
        c = self.c
        if not tracking:
            if self.state in (LOCKED, DEGRADED):
                self.state, self._lost_since = LOST, t
            elif self.state == LOST and t - self._lost_since > c.lost_to_searching_s:
                self.state = SEARCHING
            self._good_since = None
            return self.state
        if self.state in (SEARCHING, LOST):
            self.state = DEGRADED                      # new track: never jump straight to LOCKED
        good = conf >= c.conf_lock and self.quality >= c.q_lock and self.recent_valid(t)
        if self.state == DEGRADED:
            if good:
                if self._good_since is None:
                    self._good_since = t
                if t - self._good_since >= c.lock_dwell_s:
                    self.state = LOCKED
            else:
                self._good_since = None
        elif self.state == LOCKED:
            bad = (conf < c.conf_lock - c.hysteresis or self.quality < c.q_lock - c.hysteresis
                   or not self.recent_valid(t))
            if bad:
                self.state, self._good_since = DEGRADED, None
        return self.state
