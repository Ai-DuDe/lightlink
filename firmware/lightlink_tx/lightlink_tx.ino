// Light Link P1 transmitter (ESP32). Blinks a light with Manchester on-off keying.
//
// Serial (115200 baud, newline terminated):
//   msg <text>   message to send (max 32 bytes)       rate <bps>   bit rate 2..15
//   gap <ms>     dark time between frames             cal          steady 1 Hz blink for camera tests
//   run          back to sending frames               status       show settings
//   timing       timer accuracy report (hardware self-test)   timing reset
//
// Bit timing comes from a hardware timer interrupt, never delay(), so the bit period does not
// wobble with Wi-Fi, serial or loop() activity.
#include <Arduino.h>
#include "protocol.h"

// ---- hardware config -------------------------------------------------------------------
// GPIO4 is the bright white flash LED on an ESP32-CAM: no wiring needed for a first test.
// For an external LED use e.g. GPIO13 with a 220 ohm resistor to GND.
// A laser module must be switched through a transistor (see README), never straight off a pin.
#define TX_PIN 4
#define ACTIVE_HIGH 1          // set to 0 for the red status LED on GPIO33 (active low)

// ---- defaults ---------------------------------------------------------------------------
static const float    DEFAULT_RATE = 8.0f;
static const uint32_t DEFAULT_GAP_MS = 1000;
static const char    *DEFAULT_MSG = "HELLO";
static const float    MIN_RATE = 2.0f, MAX_RATE = 15.0f;
static const uint32_t CAL_HALF_PERIOD_US = 500000;   // 1 Hz: 0.5 s on, 0.5 s off

enum Mode : uint8_t { MODE_FRAME, MODE_GAP, MODE_CAL };

static volatile Mode     mode = MODE_FRAME;
static uint8_t           frame[ll::MAX_FRAME_BYTES];
static volatile size_t   frameBytes = 0;
static volatile size_t   symIdx = 0, symTotal = 0;
static volatile uint32_t gapLeft = 0, gapSymbols = 0;
static volatile uint32_t calCount = 0, calHalfSyms = 1;
static volatile uint8_t  calLevel = 0;

// timing self-test, filled by the ISR
static volatile uint32_t lastUs = 0, minDt = 0xFFFFFFFF, maxDt = 0, ticks = 0;

static hw_timer_t *timer = nullptr;
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

static char     message[ll::MAX_PAYLOAD + 1];
static float    bitRate = DEFAULT_RATE;
static uint32_t gapMs = DEFAULT_GAP_MS;
static uint32_t symbolUs() { return (uint32_t)(500000.0f / bitRate + 0.5f); }

static inline void drive(uint8_t on) { digitalWrite(TX_PIN, (on ? 1 : 0) == ACTIVE_HIGH ? HIGH : LOW); }

void IRAM_ATTR onTick() {
  portENTER_CRITICAL_ISR(&mux);
  uint32_t now = micros();
  if (lastUs) { uint32_t d = now - lastUs; if (d < minDt) minDt = d; if (d > maxDt) maxDt = d; }
  lastUs = now; ticks++;
  switch (mode) {
    case MODE_FRAME:
      drive(ll::symbol_at(frame, symIdx));
      if (++symIdx >= symTotal) { mode = MODE_GAP; gapLeft = gapSymbols; }
      break;
    case MODE_GAP:
      drive(0);
      if (gapLeft == 0 || --gapLeft == 0) { symIdx = 0; mode = MODE_FRAME; }
      break;
    case MODE_CAL:
      if (++calCount >= calHalfSyms) { calCount = 0; calLevel ^= 1; }
      drive(calLevel);
      break;
  }
  portEXIT_CRITICAL_ISR(&mux);
}

static void startTimer() {
  const uint32_t us = symbolUs();
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  if (timer) timerEnd(timer);
  timer = timerBegin(1000000);                  // 1 MHz tick = 1 us resolution
  timerAttachInterrupt(timer, &onTick);
  timerAlarm(timer, us, true, 0);
#else
  if (timer) { timerAlarmDisable(timer); timerEnd(timer); }
  timer = timerBegin(0, 80, true);              // 80 MHz / 80 = 1 MHz
  timerAttachInterrupt(timer, &onTick, true);
  timerAlarmWrite(timer, us, true);
  timerAlarmEnable(timer);
#endif
}

// Rebuild frame + timing from the current settings and restart cleanly.
static void apply(bool calibrate) {
  uint8_t tmp[ll::MAX_FRAME_BYTES];
  size_t n = ll::build_frame((const uint8_t *)message, strlen(message), tmp);
  const uint32_t us = symbolUs();
  portENTER_CRITICAL(&mux);
  memcpy(frame, tmp, n);
  frameBytes = n; symTotal = ll::symbol_count(n); symIdx = 0;
  gapSymbols = (uint32_t)(((uint64_t)gapMs * 1000 + us / 2) / us);
  calHalfSyms = max<uint32_t>(1, CAL_HALF_PERIOD_US / us); calCount = 0; calLevel = 0;
  mode = calibrate ? MODE_CAL : MODE_FRAME;
  minDt = 0xFFFFFFFF; maxDt = 0; lastUs = 0; ticks = 0;
  portEXIT_CRITICAL(&mux);
  startTimer();
}

static void status() {
  Serial.printf("OK mode=%s msg=\"%s\" (%u bytes) rate=%.2f bps symbol=%u us gap=%u ms frame=%u s\n",
                mode == MODE_CAL ? "cal" : "run", message, (unsigned)strlen(message), bitRate,
                (unsigned)symbolUs(), (unsigned)gapMs,
                (unsigned)((symTotal * (uint64_t)symbolUs() + 999999) / 1000000));
}

static void handle(String line) {
  line.trim();
  if (!line.length()) return;
  String cmd = line, arg = "";
  int sp = line.indexOf(' ');
  if (sp > 0) { cmd = line.substring(0, sp); arg = line.substring(sp + 1); }
  bool cal = (mode == MODE_CAL);
  if (cmd == "msg") {
    if (arg.length() == 0 || arg.length() > ll::MAX_PAYLOAD) { Serial.println("ERR msg needs 1..32 bytes"); return; }
    strncpy(message, arg.c_str(), ll::MAX_PAYLOAD); message[ll::MAX_PAYLOAD] = 0;
    apply(cal); status();
  } else if (cmd == "rate") {
    float r = arg.toFloat();
    if (r < MIN_RATE || r > MAX_RATE) { Serial.println("ERR rate must be 2..15"); return; }
    bitRate = r; apply(cal); status();
  } else if (cmd == "gap") {
    long g = arg.toInt();
    if (g < 0 || g > 60000) { Serial.println("ERR gap 0..60000 ms"); return; }
    gapMs = g; apply(cal); status();
  } else if (cmd == "cal") { apply(true); status();
  } else if (cmd == "run") { apply(false); status();
  } else if (cmd == "status") { status();
  } else if (cmd == "timing") {
    if (arg == "reset") { portENTER_CRITICAL(&mux); minDt = 0xFFFFFFFF; maxDt = 0; lastUs = 0; ticks = 0; portEXIT_CRITICAL(&mux); Serial.println("OK"); return; }
    portENTER_CRITICAL(&mux); uint32_t mn = minDt, mx = maxDt, tk = ticks; portEXIT_CRITICAL(&mux);
    Serial.printf("OK ticks=%u expected=%u us min=%u us max=%u us spread=%d us\n", (unsigned)tk,
                  (unsigned)symbolUs(), (unsigned)mn, (unsigned)mx, (int)(mx - mn));
  } else {
    Serial.println("ERR commands: msg <text> | rate <bps> | gap <ms> | cal | run | status | timing [reset]");
  }
}

void setup() {
  pinMode(TX_PIN, OUTPUT);
  drive(0);
  Serial.begin(115200);
  strncpy(message, DEFAULT_MSG, ll::MAX_PAYLOAD); message[ll::MAX_PAYLOAD] = 0;
  apply(false);
  Serial.println("Light Link TX ready. Type a command (msg/rate/gap/cal/run/status/timing).");
  status();
}

void loop() {
  static String buf;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { handle(buf); buf = ""; }
    else if (buf.length() < 80) buf += c;
  }
}
