import numpy as np

from lightlink import protocol
from lightlink.config import load_config
from lightlink.simulator import Simulator, TxLight


def test_light_level_follows_manchester():
    tx = TxLight(b"A", 8.0, 1.0, start_s=0.0)
    Ts = tx.Ts
    syms = protocol.frame_symbols(b"A")
    t = (np.arange(len(syms)) + 0.5) * Ts
    assert np.array_equal(tx.level(t), np.array(syms, dtype=np.float32))
    assert tx.level(np.array([-1.0]))[0] == 0.0           # dark before start
    gap_t = len(syms) * Ts + 0.5
    assert tx.level(np.array([gap_t]))[0] == 0.0          # dark in the gap


def test_blocked_interval_is_dark():
    tx = TxLight(b"A", 8.0, 1.0, 0.0, blocks=[(0.0, 100.0)])
    assert tx.level(np.linspace(0, 5, 200)).max() == 0.0


def test_frames_have_drops_and_jittered_timestamps():
    cfg = load_config(overrides={"sim": {"duration_s": 6, "drop_prob": 0.1}})
    sim = Simulator(cfg.sim, 8.0, 1.0)
    frames = list(sim.frames())
    assert 120 < len(frames) < 180            # 180 nominal, ~10 % dropped
    dts = np.diff([f.t for f in frames])
    assert dts.max() > 1.7 / 30               # at least one visible gap
    assert frames[0].image.shape == (360, 640) and frames[0].image.dtype == np.uint8


def test_spot_is_brighter_when_on_and_lamp_is_static():
    cfg = load_config(overrides={"sim": {"duration_s": 4, "noise_sigma": 0.5,
                                         "drift_amp": 0, "drop_prob": 0}})
    sim = Simulator(cfg.sim, 4.0, 1.0)
    sx, sy = int(0.62 * 640), int(0.41 * 360)
    lx, ly = int(0.25 * 640), int(0.3 * 360)
    spot, lamp = [], []
    for f in sim.frames():
        spot.append(int(f.image[sy, sx]))
        lamp.append(int(f.image[ly, lx]))
    assert max(spot) - min(spot) > 60
    assert max(lamp) - min(lamp) < 8
