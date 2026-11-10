# Light Link

Send text with a blinking light and read it with a webcam. A phone flashlight (or an ESP32 with an LED)
blinks a message; a laptop webcam finds the light, tracks it, decodes the message and shows a live dashboard.
No Wi-Fi or Bluetooth between the two devices: the only connection is light.

A message is shown as **VALID** only if its CRC checksum passes. Anything else is logged as failed and never displayed.

## Which file do I run?

| I want to... | Run |
|---|---|
| **Receive messages with my webcam** (the main thing) | `python run.py` |
| Try it with no hardware (simulated camera and light) | `python scripts/run_replay.py --simulate` |
| Check my webcam is good enough | `python scripts/camera_check.py` |
| Send from a phone | open <https://ai-dude.github.io/lightlink/> |
| Send from an ESP32 | upload `firmware/lightlink_tx/lightlink_tx.ino` |
| Run the tests | `python -m pytest -q` |

The sender and receiver must use the same bit rate (default 4). Everything else is configuration in `config.yaml`.

## What you need

- Python 3.11+ and a webcam (a 720p laptop camera is fine)
- A transmitter, either:
  - **a phone**: open the web page <https://ai-dude.github.io/lightlink/> (Android Chrome can blink the real flashlight; any phone can flash a screen square), or
  - **an ESP32** with an LED (`firmware/`)
- A dim room, but not a dark one (see Troubleshooting)

## Quick start

```bash
git clone https://github.com/Ai-DuDe/lightlink.git
cd lightlink
pip install -r requirements.txt
```

**Try it with no hardware** (simulated camera and light):

```bash
python scripts/run_replay.py --simulate --no-window
```

**Live, with a phone as the transmitter:**

1. Phone: open <https://ai-dude.github.io/lightlink/>, set bit rate **4**, tap **Calibration**.
2. Laptop: `python run.py --bit-rate 4`
3. Hold the phone 1-2 m from the laptop, light facing the camera. A box should appear around it.
4. Phone: tap **Send message**. It blinks a lead-in for 4 s so the laptop can lock on, sends the message once, then offers **Play again**. The message appears in the dashboard when it finishes.

Dashboard: click the `-` / `+` buttons (or press `-` / `+`) to change the bit rate to match the sender. Keys: `d` dark view (darkens the picture so the light stands out; display only), `l` log to CSV, `v` record raw camera video, `f` fullscreen, `q` quit. The window can be resized or maximised and the layout adapts. Output goes to `logs/`.

The phone page makes no network requests at all (its Content-Security-Policy forbids them), so what the laptop
receives has travelled only as light. The page is `docs/index.html`.

## Phone flashlight mode

Browsers only allow torch control on a secure (https) page. The link above is https, so choose **Flashlight** and
allow the camera prompt (Android Chrome only; iPhones cannot do it, use Screen mode there).

## Using an ESP32

1. Arduino IDE with the **esp32** board package, board **AI Thinker ESP32-CAM** (or your ESP32).
2. Open `firmware/lightlink_tx/lightlink_tx.ino`, set `TX_PIN` (default GPIO4, the onboard flash LED), upload.
3. Serial monitor, 115200 baud: `rate 4`, `msg HELLO`, `cal` (1 Hz test blink), `run`, `timing` (timer self-test).

External LED: GPIO -> 220 ohm resistor -> LED -> GND. Switch anything bigger than an LED through a transistor.

## Choosing a bit rate

The camera needs about 3 frames per half-bit: `max bit rate = fps / 6`. A 30 fps webcam supports about 4-5 bit/s.
Set the same rate on the transmitter and the receiver (`--bit-rate`). Short messages decode sooner:
a 2-byte message takes 14 s per frame at 4 bit/s, and the first frame after start-up is usually missed while the
receiver finds the light.

## Check your camera

```bash
python scripts/camera_check.py --cover-test
```

Reports the real frame rate, whether exposure stays fixed, and whether the camera is suitable.

## Troubleshooting

- **Very low FPS (around 5):** the room is too dark, so the webcam uses long exposures. Turn on a dim lamp.
  On Windows, `camera.backends` in `config.yaml` tries Media Foundation first (5x faster than DirectShow on the dev laptop).
- **No box around the light:** a full-screen flash is rejected as a scene change. Use the small square (default) or the flashlight.
- **State stays DEGRADED:** no valid frame yet. Check both bit rates match and the light is not blown out in the picture.

## How it works

`lightlink/` is the receiver: `detector` finds the blinking source by frame-difference energy (a steady lamp is ignored),
`tracker` follows it with a Kalman filter, `demod` thresholds the brightness and decodes Manchester bits against the
preamble and sync word, `link` scores quality and runs the SEARCHING / LOCKED / DEGRADED / LOST states.
The frame format is in [`protocol/FRAME_SPEC.md`](protocol/FRAME_SPEC.md). The receiver also outputs the spot's normalised
offset from the image centre (`dx`, `dy`) for future pan-tilt control. All settings are in `config.yaml`.

## Tests

```bash
python -m pytest -q
```

The C++ firmware check needs a compiler (g++, clang++ or `pip install ziglang`) and is skipped without one.
Optional: `python scripts/install_hooks.py` adds a pre-commit hook that blocks commits containing emails, LAN IPs,
local paths or tokens.

## Safety

Use an LED. If you ever use a laser: visible light only, **under 1 mW**, aimed at a white card, never at eyes, people,
vehicles or windows. Never infrared. Hobby "5 mW" laser modules (e.g. KY-008) are above that limit.
