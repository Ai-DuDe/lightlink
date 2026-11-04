import numpy as np

from lightlink import protocol as P
from lightlink.config import load_config
from lightlink.demod import Demodulator
from lightlink.simulator import TxLight


def _series(payload, rate, fps=30.0, seconds=None, noise=3.0, jitter=0.002, ppm=3000,
            start=1.0, seed=0, flip=None):
    rng = np.random.default_rng(seed)
    tx = TxLight(payload, rate, 1.0, start, ppm)
    seconds = seconds or (start + tx.frame_period_s + 1.5)
    t = np.arange(0, seconds, 1 / fps) + rng.normal(0, jitter, int(seconds * fps) + 1)[:int(np.ceil(seconds * fps))]
    t = np.sort(t)
    lvl = np.array([tx.level_exposed(x, 0.5 / fps) for x in t])
    if flip is not None:                                   # corrupt one symbol pair region
        a = start + flip * 2 * tx.Ts
        lvl[(t >= a) & (t < a + 2 * tx.Ts)] = 1 - lvl[(t >= a) & (t < a + 2 * tx.Ts)]
    v = 40 + 100 * lvl + rng.normal(0, noise, len(t))
    return t, v


def _decode(t, v, rate):
    cfg = load_config()
    d = Demodulator(cfg.demod, rate)
    events = []
    for ti, vi in zip(t, v):
        d.push(ti, vi)
        events += d.poll(ti)
    return events


def test_decodes_clean_frame_with_clock_error_and_jitter():
    t, v = _series(b"HI", 8.0)
    ev = _decode(t, v, 8.0)
    assert [e.payload for e in ev if e.valid] == [b"HI"]


def test_decodes_at_low_rate_max_payload():
    payload = bytes(range(32))
    t, v = _series(payload, 4.0, fps=30.0)
    assert [e.payload for e in _decode(t, v, 4.0) if e.valid] == [payload]


def test_corrupted_frame_is_failed_never_valid():
    # flip bits mid-payload by inverting a whole bit's light: CRC must catch it
    t, v = _series(b"HELLO", 8.0, flip=8 + 16 + 8 + 12)
    ev = _decode(t, v, 8.0)
    assert not any(e.valid for e in ev)
    assert any((not e.valid) and e.payload is None for e in ev)


def test_noise_only_produces_no_events():
    rng = np.random.default_rng(5)
    t = np.arange(0, 60, 1 / 30)
    v = 50 + rng.normal(0, 3, len(t))
    assert _decode(t, v, 8.0) == []


def test_random_blinking_produces_no_valid_frame():
    rng = np.random.default_rng(6)
    t = np.arange(0, 60, 1 / 30)
    v = 50 + 80 * (rng.random(len(t)) > 0.5) + rng.normal(0, 3, len(t))
    assert not any(e.valid for e in _decode(t, v, 8.0))


def test_repeated_frames_decoded_once_each():
    t, v = _series(b"HI", 8.0, seconds=1.0 + 3 * (P.frame_duration_s(2, 8.0) + 1.0) + 0.5)
    valid = [e for e in _decode(t, v, 8.0) if e.valid]
    assert len(valid) == 3
