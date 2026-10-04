#include "protocol.h"

#include <iostream>
using namespace rtx_cuda;
int main() {
  try {
    for (size_t n = 0; n != 1024; ++n) {
      std::vector<uint8_t> bytes(n);
      for (size_t i = 0; i < n; ++i) {
        bytes[i] = static_cast<uint8_t>(i * 73 + n);
      }
      Require(Decode(Encode(bytes), n) == bytes, "Base64 roundtrip failure");
    }
    for (const auto* input :
         {"=AAA", "A===", "AA=A", "AB==", "AAB=", "AA==AAAA", "AA?=", "A"}) {
      bool rejected = false;
      try {
        Decode(input, 64);
      } catch (...) {
        rejected = true;
      }
      Require(rejected, "Malformed base64 accepted");
    }
    for (const auto& input :
         {Json(-1), Json(1.5), Json("2"), Json(uint64_t(-1))}) {
      bool rejected = false;
      try {
        UInt(input, 0, 100);
      } catch (...) {
        rejected = true;
      }
      Require(rejected, "Invalid integer accepted");
    }
    std::cout << "1024 binary roundtrips and malformed input cases passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what();
    return 1;
  }
}
