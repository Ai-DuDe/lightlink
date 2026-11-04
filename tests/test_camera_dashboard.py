import cv2
import numpy as np

from lightlink.camera import Camera, fps_stats, max_bit_rate
from lightlink.config import load_config
from lightlink.dashboard import DashboardState, render
from lightlink.receiver import Receiver
from lightlink.simulator import Simulator


def test_fps_stats_detects_drops_and_rate():
    t = np.arange(0, 3, 1 / 30)
    t = np.delete(t, [20, 21, 50])                      # three dropped frames
    st = fps_stats(t)
    assert abs(st["fps"] - 30) < 0.5
    assert 2.5 < st["dropped_pct"] < 4.5


def test_max_bit_rate_from_fps():
    assert max_bit_rate(30, 3.0) == 5.0
    assert max_bit_rate(120, 3.0) == 15.0               # capped at the protocol ceiling


class FakeCap:
    """Camera that ignores everything (like many cheap webcams) or obeys everything."""

    def __init__(self, obey):
        self.obey, self.p = obey, {cv2.CAP_PROP_AUTO_EXPOSURE: 0.75, cv2.CAP_PROP_EXPOSURE: -3.0,
                                   cv2.CAP_PROP_GAIN: 40.0}

    def set(self, k, v):
        if self.obey:
            self.p[k] = v
        return True

    def get(self, k):
        return self.p.get(k, 0.0)

    def grab(self):
        return True


def _cam(obey):
    cfg = load_config()
    cam = Camera(cfg.camera)
    cam.cap = FakeCap(obey)
    return cam, cam.lock_exposure()


def test_lock_report_flags_disobedient_camera():
    _, rep = _cam(obey=False)
    assert rep.exposure_locked is False
    assert rep.obeyed["exposure"] is False and rep.obeyed["gain"] is False


def test_lock_report_accepts_obedient_camera():
    _, rep = _cam(obey=True)
    assert rep.exposure_locked is True


def test_dashboard_renders_headless():
    cfg = load_config(overrides={"sim": {"duration_s": 6, "payload": "HI"}})
    sim, rx, ds = Simulator(cfg.sim, 8.0, 1.0), Receiver(cfg), DashboardState(cfg.dashboard)
    for f in sim.frames():
        u = rx.process(f.t, f.image)
        ds.add(u)
    out = render(f.image, u, ds, 8.0, True, True)
    assert out.ndim == 3 and out.shape[1] == 640 + 380 and out.std() > 5
