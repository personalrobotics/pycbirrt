// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <string>
#include <vector>

#include "sscbirrt/types.hpp"

namespace sscbirrt {

// State validity (collision, in practice). Applied to every root and every configuration along every edge.
class StateValidator {
 public:
  virtual ~StateValidator() = default;
  virtual bool is_valid(ConfigView q) const = 0;
};

class AcceptAll final : public StateValidator {
 public:
  bool is_valid(ConfigView) const override { return true; }
};

// Invalid strictly inside any listed axis-aligned box in joint space (open on every side, as
// sscbirrt.testing.Wall is). Infinite bounds are allowed, so a Wall(axis, lo, hi) is a box that
// is unbounded on the other axes.
class JointBoxObstacles final : public StateValidator {
 public:
  struct Box {
    std::vector<double> lo, hi;
  };
  explicit JointBoxObstacles(std::vector<Box> boxes);
  bool is_valid(ConfigView q) const override;

 private:
  std::vector<Box> boxes_;
};

}  // namespace sscbirrt
