// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
//
// SHA-256 (FIPS 180-4), standard library only, for provenance hashes. Not for security.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace sscbirrt::detail {

class Sha256 {
 public:
  Sha256();
  void update(std::span<const std::byte> data);
  void update(const void* data, std::size_t n) { update(std::span<const std::byte>(static_cast<const std::byte*>(data), n)); }
  std::array<std::uint8_t, 32> digest();  // finalizes; the object must not be updated afterward
  static std::string hex(std::span<const std::byte> data);  // one-shot lowercase hex digest

 private:
  void block(const std::uint8_t* p);
  std::array<std::uint32_t, 8> h_;
  std::array<std::uint8_t, 64> buf_{};
  std::size_t buf_len_ = 0;
  std::uint64_t total_ = 0;
};

}  // namespace sscbirrt::detail
