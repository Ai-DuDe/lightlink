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
    # any window size, landscape / portrait / tiny / 4K-ish, must render at exactly that size
    for w, h in [(1280, 720), (1920, 1080), (640, 480), (900, 1100), (320, 240), (3000, 1600)]:
        out = render(f.image, u, ds, 8.0, True, True, size=(w, h))
        assert out.shape == (h, w, 3) and out.std() > 5, (w, h)
    out = render(f.image, u, ds, 8.0, False, False, size=(1280, 720))
    assert set(ds.buttons) == {'rate_down', 'rate_up', 'dark'}   # clickable buttons are laid out


def test_dark_view_blacks_out_background_but_keeps_the_light():
    import numpy as np
    from lightlink.dashboard import _dark_view
    frame = np.full((360, 640, 3), 110, np.uint8)          # a bright, washed-out room
    frame[170:176, 300:306] = 250                           # the blinking light
    out = _dark_view(frame)
    assert out[50, 50].max() < 15                           # background pushed to black
    assert out[172, 302].min() > 200                        # light still bright
    assert (frame[50, 50] == 110).all()                     # the original frame is untouched
