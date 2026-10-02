// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <array>
#include <optional>
#include <string>

namespace sscbirrt {

// A 4x4 homogeneous transform, row-major. The core stays free of Eigen and of sstsr; the pose-region math is
// sstsr's own C++ behind sscbirrt::tsr (#184), and this is what the core and the kinematics adapters need.
struct Transform {
  std::array<double, 16> m{};

  static Transform identity();
  static Transform from_rows(const std::array<std::array<double, 4>, 4>& rows);
  double at(int r, int c) const { return m[static_cast<std::size_t>(r * 4 + c)]; }
  double& at(int r, int c) { return m[static_cast<std::size_t>(r * 4 + c)]; }
  Transform operator*(const Transform& o) const;
  Transform inverse_rigid() const;  // for a rigid transform: transpose the rotation, negate-rotate the translation
  bool operator==(const Transform& o) const { return m == o.m; }
};

// sstsr 3.2.0's construction contract (_check_frame): finite, last row 0 0 0 1, R R^T = I and det R = +1
// within FRAME_ATOL. Returns the violation, or nullopt.
constexpr double kFrameAtol = 1e-6;  // tsr.FRAME_ATOL
std::optional<std::string> why_not_frame(const Transform& T, const std::string& name);

}  // namespace sscbirrt
