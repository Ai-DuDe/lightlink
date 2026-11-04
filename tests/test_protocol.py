import json
import subprocess
import sys
from pathlib import Path

import pytest

from lightlink import protocol as P

ROOT = Path(__file__).resolve().parent.parent


def test_crc8_check_value():
    assert P.crc8(b"123456789") == 0xF4


@pytest.mark.parametrize("payload", [b"", b"A", b"HELLO", bytes(range(32))])
def test_roundtrip(payload):
    res = P.parse_frame(P.build_frame(payload))
    assert res.ok and res.payload == payload


def test_manchester_roundtrip_and_convention():
    assert P.manchester_encode([1, 0]) == [1, 0, 0, 1]
    bits = P.frame_bits(b"xyz")
    assert P.manchester_decode(P.manchester_encode(bits)) == bits


def test_manchester_rejects_missing_transition():
    with pytest.raises(ValueError):
        P.manchester_decode([1, 1, 0, 1])


def test_payload_too_long():
    with pytest.raises(ValueError):
        P.build_frame(bytes(33))


def test_every_single_bit_flip_is_rejected():
    frame = bytearray(P.build_frame(b"HELLO"))
    body_start_bit = P.PREAMBLE_BITS + P.SYNC_BITS
    bits = P.bytes_to_bits(bytes(frame))
    for i in range(body_start_bit, len(bits)):
        bad = list(bits)
        bad[i] ^= 1
        res = P.parse_body_bits(bad[body_start_bit:])
        assert not res.ok and res.payload is None, f"flip at {i} accepted"


def test_bad_length_rejected_without_payload():
    bits = P.bytes_to_bits(bytes([200]) + bytes(10))
    res = P.parse_body_bits(bits)
    assert not res.ok and res.reason == "bad_length" and res.payload is None


def test_preamble_is_clock_tone():
    sym = P.frame_symbols(b"")[:16]
    assert sym == [1, 0] * 8


def test_vectors_match_python():
    data = json.loads((ROOT / "protocol" / "vectors.json").read_text())
    assert data["crc8_check"]["crc"] == P.crc8(b"123456789")
    for v in data["vectors"]:
        p = bytes.fromhex(v["payload_hex"])
        assert P.build_frame(p).hex() == v["frame_hex"]
        assert "".join(map(str, P.frame_symbols(p))) == v["symbols"]
