// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <array>
#include <optional>
#include <string>

namespace sscbirrt {

// A 4x4 homogeneous transform, row-major. The core stays free of Eigen; this is the
// handful of operations the pose-region math needs (docs/native-design.md, v1.6.0).
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

using Rpy = std::array<double, 3>;
using XyzRpy = std::array<double, 6>;

// sstsr's conventions (tsr/core/tsr.py). rpy_to_rot is Z-Y-X: yaw about z, then pitch about y, then
// roll about x. rot_to_rpy takes pitch = -asin(R[2,0]) off the gimbal lock (|R[2,0]| within 1e-9 of 1)
// and the coupled branches at the lock with yaw fixed to 0.
std::array<double, 9> rpy_to_rot(const Rpy& rpy);           // row-major 3x3
Rpy rot_to_rpy(const Transform& T);                          // from T's rotation block
Transform xyzrpy_to_trans(const XyzRpy& v);
XyzRpy trans_to_xyzrpy(const Transform& T);

// Wrap each angle into [lower, lower + 2pi), as sstsr's wrap_to_interval (with its guard against
// a modulo result rounding up to exactly 2pi).
double wrap_to_interval(double angle, double lower);

// sstsr 3.2.0's construction contract (_check_frame): finite, last row 0 0 0 1, R R^T = I and det R = +1
// within FRAME_ATOL. Returns the violation, or nullopt.
constexpr double kFrameAtol = 1e-6;  // tsr.FRAME_ATOL
std::optional<std::string> why_not_frame(const Transform& T, const std::string& name);

}  // namespace sscbirrt
