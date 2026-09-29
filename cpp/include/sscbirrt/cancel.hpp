// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>

namespace sscbirrt {

// Cooperative cancellation. Polled once per search iteration, before each root draw, and
// before each smoothing attempt; a set or validator that runs long is not interrupted.
class CancellationToken {
 public:
  void cancel() noexcept { flag_.store(true, std::memory_order_relaxed); }
  bool cancelled() const noexcept { return flag_.load(std::memory_order_relaxed); }

 private:
  std::atomic<bool> flag_{false};
};

struct SolveOptions {
  std::optional<std::uint64_t> seed;                // nullopt: seeded from std::random_device
  std::shared_ptr<const CancellationToken> cancel;  // may be null
  bool keep_trees = true;                           // attach the trees to the result for inspection
};

}  // namespace sscbirrt
