// Light Link frame codec. Mirrors protocol/FRAME_SPEC.md and lightlink/protocol.py.
// Header-only and free of Arduino dependencies so tests can compile it on a PC.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace ll {

constexpr uint8_t  PREAMBLE_BYTE = 0xFF;    // all ones: Manchester square wave at the bit rate
constexpr uint16_t SYNC_WORD     = 0x2DD4;
constexpr size_t   MAX_PAYLOAD   = 32;
constexpr size_t   MAX_FRAME_BYTES = 1 + 2 + 1 + MAX_PAYLOAD + 1;
constexpr uint8_t  CRC_POLY      = 0x07;    // init 0x00, no reflection, no final xor

inline uint8_t crc8(const uint8_t *d, size_t n) {
  uint8_t crc = 0x00;
  for (size_t i = 0; i < n; i++) {
    crc ^= d[i];
    for (int b = 0; b < 8; b++) crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ CRC_POLY) : (uint8_t)(crc << 1);
  }
  return crc;
}

// Writes preamble, sync, length, payload, CRC into out (>= MAX_FRAME_BYTES). Returns the byte
// count, or 0 if the payload is too long.
inline size_t build_frame(const uint8_t *payload, size_t n, uint8_t *out) {
  if (n > MAX_PAYLOAD) return 0;
  size_t k = 0;
  out[k++] = PREAMBLE_BYTE;
  out[k++] = (uint8_t)(SYNC_WORD >> 8);
  out[k++] = (uint8_t)(SYNC_WORD & 0xFF);
  const size_t body = k;                   // CRC covers length byte + payload
  out[k++] = (uint8_t)n;
  for (size_t i = 0; i < n; i++) out[k++] = payload[i];
  out[k] = crc8(out + body, k - body);
  return k + 1;
}

// Manchester, MSB first: bit 1 -> ON,OFF ; bit 0 -> OFF,ON. Returns 1 for light on.
// Computed on the fly so the timer interrupt needs no big symbol buffer.
inline uint8_t symbol_at(const uint8_t *frame, size_t sym) {
  const size_t bit = sym >> 1;
  const uint8_t v = (frame[bit >> 3] >> (7 - (bit & 7))) & 1;
  return (sym & 1) ? (uint8_t)!v : v;
}

inline size_t symbol_count(size_t frame_bytes) { return frame_bytes * 16; }

}  // namespace ll
