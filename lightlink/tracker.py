"""Constant-velocity Kalman tracker for the spot, plus an honest confidence number."""
import math

import numpy as np


class SpotTracker:
    def __init__(self, cfg):
        self.c = cfg
        self.active = False
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.t = 0.0
        self.last_meas_t = 0.0
        self.reject_ema = 0.0        # share of recent frames where a far-away source competed

    def start(self, x, y, t):
        r2 = self.c.meas_noise_px ** 2
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([r2, r2, 20.0 ** 2, 20.0 ** 2])
        self.t = self.last_meas_t = t
        self.active = True
        self.reject_ema = 0.0

    def drop(self):
        self.active = False

    def predict(self, t):
        dt = max(t - self.t, 0.0)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.c.process_noise
        Q = np.zeros((4, 4))
        for i in (0, 1):  # discrete white-noise acceleration on each axis
            Q[i, i], Q[i, i + 2], Q[i + 2, i], Q[i + 2, i + 2] = \
                q * dt ** 4 / 4, q * dt ** 3 / 2, q * dt ** 3 / 2, q * dt ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, meas, rejected: bool = False):
        """meas = (x, y) or None. Call predict(t) first."""
        self.reject_ema += 0.1 * ((1.0 if rejected else 0.0) - self.reject_ema)
        if meas is None:
            return
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        R = np.eye(2) * self.c.meas_noise_px ** 2
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(meas) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.last_meas_t = self.t

    @property
    def pos(self):
        return float(self.x[0]), float(self.x[1])

    def silence(self, t) -> float:
        return max(0.0, t - self.last_meas_t)

    def confidence(self, t) -> float:
        """1 when measurements arrive; decays while blind or when rivals keep appearing."""
        if not self.active:
            return 0.0
        sigma = math.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0) / 2)
        pos_term = max(0.0, 1.0 - sigma / self.c.max_pos_sigma_px)
        return math.exp(-self.silence(t) / self.c.conf_tau_s) * pos_term * (1.0 - self.reject_ema)
