// Host-side driver: reads one hex payload per line ("-" = empty) and prints
// "<frame hex> <symbol string>" using the firmware's own protocol.h.
#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
#include "../lightlink_tx/protocol.h"

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    uint8_t payload[ll::MAX_PAYLOAD + 8];
    size_t n = 0;
    if (line != "-") {
      for (size_t i = 0; i + 1 < line.size(); i += 2) payload[n++] = (uint8_t)std::stoi(line.substr(i, 2), nullptr, 16);
    }
    uint8_t frame[ll::MAX_FRAME_BYTES];
    size_t fb = ll::build_frame(payload, n, frame);
    for (size_t i = 0; i < fb; i++) std::printf("%02x", frame[i]);
    std::printf(" ");
    for (size_t s = 0; s < ll::symbol_count(fb); s++) std::printf("%d", ll::symbol_at(frame, s));
    std::printf("\n");
  }
  std::printf("crc %02x\n", ll::crc8((const uint8_t *)"123456789", 9));
  return 0;
}
