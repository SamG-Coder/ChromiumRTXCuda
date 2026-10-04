// Copyright 2026 The ChromiumRTXCuda Authors. BSD-3-Clause; see ../LICENSE.
#pragma once
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

namespace rtx_cuda {
using Json = nlohmann::json;
inline constexpr uint32_t kMaxMessage = 900 * 1024;
inline constexpr size_t kMaxReadback = 512 * 1024;
inline constexpr size_t kMaxAllocation = 64 * 1024 * 1024;

inline void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

inline uint32_t UInt(const Json& j, uint32_t low, uint32_t high) {
  Require(j.is_number_unsigned() || j.is_number_integer(), "Expected integer");
  if (j.is_number_integer() && !j.is_number_unsigned()) {
    Require(j.get<int64_t>() >= 0, "Negative integer");
  }
  const auto v = j.get<uint64_t>();
  Require(v >= low && v <= high, "Integer outside allowed range");
  return static_cast<uint32_t>(v);
}

inline std::string Encode(std::span<const uint8_t> bytes) {
  constexpr char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  result.reserve((bytes.size() + 2) / 3 * 4);
  for (size_t i = 0; i < bytes.size(); i += 3) {
    uint32_t n = uint32_t(bytes[i]) << 16;
    if (i + 1 < bytes.size()) {
      n |= uint32_t(bytes[i + 1]) << 8;
    }
    if (i + 2 < bytes.size()) {
      n |= bytes[i + 2];
    }
    result += alphabet[(n >> 18) & 63];
    result += alphabet[(n >> 12) & 63];
    result += i + 1 < bytes.size() ? alphabet[(n >> 6) & 63] : '=';
    result += i + 2 < bytes.size() ? alphabet[n & 63] : '=';
  }
  return result;
}

inline std::vector<uint8_t> Decode(const std::string& s, size_t maximum) {
  Require(s.size() % 4 == 0 && s.size() <= (maximum + 2) / 3 * 4,
          "Invalid base64 length");
  std::vector<uint8_t> result;
  result.reserve(s.size() / 4 * 3);
  auto digit = [](char c) -> uint32_t {
    if (c >= 'A' && c <= 'Z') {
      return c - 'A';
    }
    if (c >= 'a' && c <= 'z') {
      return c - 'a' + 26;
    }
    if (c >= '0' && c <= '9') {
      return c - '0' + 52;
    }
    if (c == '+') {
      return 62;
    }
    if (c == '/') {
      return 63;
    }
    throw std::runtime_error("Invalid base64 character");
  };
  for (size_t i = 0; i < s.size(); i += 4) {
    const bool p2 = s[i + 2] == '=', p3 = s[i + 3] == '=';
    Require(!p2 || p3, "Invalid base64 padding");
    Require(!(p2 || p3) || i + 4 == s.size(), "Early base64 padding");
    uint32_t a = digit(s[i]), b = digit(s[i + 1]);
    uint32_t c = p2 ? 0 : digit(s[i + 2]), d = p3 ? 0 : digit(s[i + 3]);
    Require((!p2 || (b & 15) == 0) && (!p3 || p2 || (c & 3) == 0),
            "Noncanonical base64 padding");
    const uint32_t n = (a << 18) | (b << 12) | (c << 6) | d;
    result.push_back(static_cast<uint8_t>(n >> 16));
    if (!p2) {
      result.push_back(static_cast<uint8_t>(n >> 8));
    }
    if (!p3) {
      result.push_back(static_cast<uint8_t>(n));
    }
  }
  Require(result.size() <= maximum, "Decoded data too large");
  return result;
}
}  // namespace rtx_cuda
