# Light Link frame specification (v1)

Single source of truth. `lightlink/protocol.py` (Python) and
`firmware/lightlink_tx/protocol.h` (C++) both implement exactly this, and
`tests/test_crosslang.py` checks them against `protocol/vectors.json`.

## Modulation

* On-off keying: light ON = 1 symbol, light OFF = 0 symbol.
* Manchester encoding, one transition in the middle of every bit:
  * data bit **1** -> symbols `ON, OFF` (`1,0`)
  * data bit **0** -> symbols `OFF, ON` (`0,1`)
* Bit rate: configurable, 2..15 bit/s (default 8). Symbol period = 1 / (2 * bit rate).
* Bits are sent most-significant bit first, bytes in order.
* Between frames the light stays OFF for a configurable gap (default 1 s).

Why Manchester: every bit has a transition, so the receiver never loses clock on a run
of identical bits, and the on/off levels are balanced so an adaptive threshold works.

## Frame

| Field     | Size            | Value                                              |
|-----------|-----------------|----------------------------------------------------|
| preamble  | 8 bits (0xFF)   | all ones -> Manchester square wave at the bit rate |
| sync word | 16 bits         | `0x2DD4`                                           |
| length    | 8 bits          | payload bytes, 0..32                               |
| payload   | length bytes    | arbitrary bytes                                    |
| CRC-8     | 8 bits          | over `length` byte + payload                       |

The preamble gives the receiver an unmistakable clock tone. The sync word starts with
a `0` bit, which breaks the tone (two consecutive OFF symbols), marking the frame start.

## CRC-8

Polynomial `x^8 + x^2 + x + 1` (0x07), init `0x00`, no reflection, no final XOR.
Check value: `crc8("123456789") == 0xF4`.

## Receiver rules

* A frame is VALID only if sync matches, `length <= 32`, and the CRC matches.
* Everything else that reached a sync word is logged as FAILED. Payload text of a
  failed frame is never shown as a message.

## Limits

Maximum frame = 8 + 16 + 8 + 256 + 8 = 296 bits: 148 s at 2 bit/s, 74 s at 4 bit/s,
37 s at 8 bit/s. Keep payloads short at low bit rates.
