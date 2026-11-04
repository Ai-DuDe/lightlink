import numpy as np

from lightlink.config import load_config
from lightlink.detector import BlinkDetector
from lightlink.simulator import Simulator
from lightlink.tracker import SpotTracker


def _run_detector(overrides, seconds):
    cfg = load_config(overrides={"sim": overrides})
    sim = Simulator(cfg.sim, 8.0, 1.0)
    det = BlinkDetector(cfg.detector, cfg.sim.width, cfg.sim.height)
    peak = None
    for f in sim.frames():
        if f.t > seconds:
            break
        det.update(f.image, f.t)
        peak = det.find_peak(f.t)
    return peak, cfg


def test_finds_blinking_spot_not_static_lamp():
    peak, cfg = _run_detector({"duration_s": 8, "start_s": 0.0}, 7)
    assert peak is not None
    x, y, _ = peak
    assert abs(x - 0.62 * 640) < 3 and abs(y - 0.41 * 360) < 3   # spot, not the lamp at (160, 108)


def test_static_lamp_alone_is_not_detected():
    # transmitter never starts inside the run: only the bright static lamp and noise exist
    peak, _ = _run_detector({"duration_s": 8, "start_s": 999.0}, 7.5)
    assert peak is None


def test_scene_wide_brightness_jump_is_rejected():
    cfg = load_config()
    det = BlinkDetector(cfg.detector, 640, 360)
    rng = np.random.default_rng(0)
    peak = None
    for i in range(150):
        level = 60 if (i // 5) % 2 == 0 else 140          # whole frame flashes
        img = np.clip(level + rng.normal(0, 3, (360, 640)), 0, 255).astype(np.uint8)
        det.update(img, i / 30)
        peak = det.find_peak(i / 30)
    assert peak is None


def test_kalman_follows_moving_spot_and_coasts():
    cfg = load_config()
    tr = SpotTracker(cfg.tracker)
    rng = np.random.default_rng(1)
    tr.start(100, 100, 0.0)
    for i in range(1, 90):
        t = i / 30
        tr.predict(t)
        tr.update((100 + 20 * t + rng.normal(0, 0.5), 100 + rng.normal(0, 0.5)))
    assert abs(tr.pos[0] - (100 + 20 * 89 / 30)) < 2
    assert abs(tr.x[2] - 20) < 4                          # velocity learned
    conf_seen = tr.confidence(89 / 30)
    t = 89 / 30 + 3.0
    tr.predict(t)
    tr.update(None)
    assert tr.confidence(t) < conf_seen * 0.6             # blind -> less confident
