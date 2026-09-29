// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <vector>

namespace sscbirrt {

using Config = std::vector<double>;          // owned configuration of length dof
using ConfigView = std::span<const double>;  // borrowed, valid for the call
using Provenance = std::vector<int>;         // choices that produced a sample, outermost first
using Rng = std::mt19937_64;                 // owned by the solver, passed by reference to sets
using Metric = std::function<double(ConfigView, ConfigView)>;

inline Config to_config(ConfigView v) { return Config(v.begin(), v.end()); }

inline double euclidean(ConfigView a, ConfigView b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = b[i] - a[i];
    s += d * d;
  }
  return std::sqrt(s);
}

// Deterministic draws that do not depend on the standard library's distribution
// implementations, so a seeded solve is repeatable across libstdc++ and libc++.
inline double unit(Rng& rng) { return static_cast<double>(rng() >> 11) * 0x1.0p-53; }  // [0, 1)

inline std::size_t index(Rng& rng, std::size_t n) {
  const auto i = static_cast<std::size_t>(unit(rng) * static_cast<double>(n));
  return i < n ? i : n - 1;
}

}  // namespace sscbirrt
