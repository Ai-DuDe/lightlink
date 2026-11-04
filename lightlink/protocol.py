"""Light Link frame codec. Spec: protocol/FRAME_SPEC.md (keep in sync with protocol.h)."""
from dataclasses import dataclass

PREAMBLE_BITS = 8          # all ones: Manchester turns this into a clean square wave
SYNC_WORD = 0x2DD4
SYNC_BITS = 16
LENGTH_BITS = 8
CRC_BITS = 8
MAX_PAYLOAD = 32
CRC_POLY = 0x07
CRC_INIT = 0x00
MIN_BIT_RATE = 2.0
MAX_BIT_RATE = 15.0
DEFAULT_BIT_RATE = 8.0


def crc8(data: bytes) -> int:
    crc = CRC_INIT
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ CRC_POLY) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def bytes_to_bits(data: bytes) -> list[int]:
    return [(b >> i) & 1 for b in data for i in range(7, -1, -1)]


def bits_to_bytes(bits) -> bytes:
    if len(bits) % 8:
        raise ValueError("bit count must be a multiple of 8")
    return bytes(
        sum(bit << (7 - i) for i, bit in enumerate(bits[n:n + 8]))
        for n in range(0, len(bits), 8)
    )


def manchester_encode(bits) -> list[int]:
    out = []
    for b in bits:
        out += (1, 0) if b else (0, 1)
    return out


def manchester_decode(symbols) -> list[int]:
    if len(symbols) % 2:
        raise ValueError("odd symbol count")
    bits = []
    for a, b in zip(symbols[0::2], symbols[1::2]):
        if (a, b) == (1, 0):
            bits.append(1)
        elif (a, b) == (0, 1):
            bits.append(0)
        else:
            raise ValueError("invalid Manchester pair")  # no mid-bit transition
    return bits


def build_frame(payload: bytes) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload > {MAX_PAYLOAD} bytes")
    body = bytes([len(payload)]) + payload
    return bytes([0xFF]) + SYNC_WORD.to_bytes(2, "big") + body + bytes([crc8(body)])


def frame_bits(payload: bytes) -> list[int]:
    return bytes_to_bits(build_frame(payload))


def frame_symbols(payload: bytes) -> list[int]:
    return manchester_encode(frame_bits(payload))


def preamble_sync_bits() -> list[int]:
    return bytes_to_bits(bytes([0xFF]) + SYNC_WORD.to_bytes(2, "big"))


def preamble_sync_symbols() -> list[int]:
    return manchester_encode(preamble_sync_bits())


def body_bits_needed(length_byte: int) -> int:
    """Bits after the sync word: length + payload + CRC."""
    return LENGTH_BITS + 8 * length_byte + CRC_BITS


def frame_duration_s(payload_len: int, bit_rate: float) -> float:
    bits = PREAMBLE_BITS + SYNC_BITS + LENGTH_BITS + 8 * payload_len + CRC_BITS
    return bits / bit_rate


@dataclass
class ParseResult:
    ok: bool
    payload: bytes | None   # only set when ok: unverified bytes are never exposed
    reason: str             # "ok", "bad_length", "bad_crc", "short"


def parse_body_bits(bits) -> ParseResult:
    """Parse the bits that follow the sync word."""
    if len(bits) < LENGTH_BITS:
        return ParseResult(False, None, "short")
    length = bits_to_bytes(bits[:LENGTH_BITS])[0]
    if length > MAX_PAYLOAD:
        return ParseResult(False, None, "bad_length")
    if len(bits) < body_bits_needed(length):
        return ParseResult(False, None, "short")
    body = bits_to_bytes(bits[:LENGTH_BITS + 8 * length])
    got = bits_to_bytes(bits[LENGTH_BITS + 8 * length:body_bits_needed(length)])[0]
    if crc8(body) != got:
        return ParseResult(False, None, "bad_crc")
    return ParseResult(True, body[1:], "ok")


def parse_frame(data: bytes) -> ParseResult:
    """Parse a whole frame given as bytes (preamble, sync, length, payload, CRC)."""
    if len(data) < 5 or data[0] != 0xFF or int.from_bytes(data[1:3], "big") != SYNC_WORD:
        return ParseResult(False, None, "no_sync")
    return parse_body_bits(bytes_to_bits(data[3:]))
