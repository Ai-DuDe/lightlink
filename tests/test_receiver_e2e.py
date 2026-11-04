import numpy as np

from lightlink.config import load_config
from lightlink.link import LOCKED, LOST, SEARCHING
from lightlink.replay import read_video, run_frames, save_video
from lightlink.simulator import Simulator


def _sim(sim=None, rate=8.0, proto=None):
    sim = {"payload": "HI", "duration_s": 30, **(sim or {})}
    cfg = load_config(overrides={"sim": sim, "protocol": {"bit_rate_bps": rate, **(proto or {})}})
    return cfg, Simulator(cfg.sim, rate, cfg.protocol.gap_s)


def test_end_to_end_valid_messages_and_lock():
    cfg, sim = _sim()
    res = run_frames(cfg, sim.frames())
    valid = [e for e in res.events if e.valid]
    assert len(valid) >= 2 and all(e.payload == b"HI" for e in valid)
    assert res.first_time(LOCKED) is not None
    # offset points at the spot: right of and above/below centre as configured (62 %, 41 %)
    _, dx, dy = res.offsets[-1]
    assert abs(dx - (0.62 * 640 - 319.5) / 319.5) < 0.03
    assert abs(dy - (0.41 * 360 - 179.5) / 179.5) < 0.05


def test_static_lamp_only_never_locks_or_decodes():
    cfg, sim = _sim({"start_s": 999.0, "duration_s": 20})
    res = run_frames(cfg, sim.frames())
    assert res.events == [] and res.first_time(LOCKED) is None
    assert all(s == SEARCHING for _, s, _ in res.timeline)


def test_block_then_recovery():
    cfg, sim = _sim({"duration_s": 60, "blocks": [[16.0, 28.0]]})
    res = run_frames(cfg, sim.frames())
    assert res.first_time(LOST, after=16.0) is not None            # 12 s dark -> track dropped
    t_lost = res.first_time(LOST, after=16.0)
    assert res.first_time(LOCKED, after=t_lost) is not None         # and it comes back
    assert res.first_time(LOCKED, after=28.0) > 28.0


def test_high_noise_never_shows_wrong_text_as_valid():
    cfg, sim = _sim({"noise_sigma": 40.0, "duration_s": 40})
    res = run_frames(cfg, sim.frames())
    assert all(e.payload == b"HI" for e in res.events if e.valid)   # valid => correct, or absent


def test_wandering_spot_is_tracked():
    cfg, sim = _sim({"wander_px": 6.0})
    res = run_frames(cfg, sim.frames())
    assert any(e.valid and e.payload == b"HI" for e in res.events)


def test_video_roundtrip_replay(tmp_path):
    cfg, sim = _sim({"duration_s": 22, "width": 320, "height": 180, "spot_sigma_px": 1.5})
    p = tmp_path / "sim.avi"
    save_video(p, sim.frames(), 30)
    res = run_frames(cfg, read_video(p))
    assert any(e.valid and e.payload == b"HI" for e in res.events)


def test_changing_bit_rate_live_fixes_a_mismatch():
    from lightlink.receiver import Receiver
    cfg, sim = _sim({"duration_s": 45}, rate=6.0)          # transmitter at 6 bit/s
    cfg = load_config(overrides={"protocol": {"bit_rate_bps": 4.0}, "sim": {"payload": "HI", "duration_s": 45}})
    rx, before, after = Receiver(cfg), [], []
    for f in sim.frames():
        if f.t > 15 and rx.bit_rate != 6.0:
            rx.set_bit_rate(6.0)                           # what the dashboard + button does
        u = rx.process(f.t, f.image)
        (after if rx.bit_rate == 6.0 else before).extend(e for e in u.events if e.valid)
    assert before == []                                    # wrong rate: nothing is decoded
    assert after and all(e.payload == b"HI" for e in after)


def test_set_bit_rate_rejects_out_of_range():
    import pytest
    from lightlink.receiver import Receiver
    rx = Receiver(load_config())
    for bad in (1.0, 16.0):
        with pytest.raises(ValueError):
            rx.set_bit_rate(bad)
