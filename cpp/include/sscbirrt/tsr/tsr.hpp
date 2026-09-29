// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <array>
#include <optional>
#include <utility>

#include "sscbirrt/transform.hpp"
#include "sscbirrt/types.hpp"

namespace sscbirrt::tsr {

constexpr double kEpsilon = 0.001;  // sstsr's utils.EPSILON: the containment slack

// [lo, hi] for x, y, z, roll, pitch, yaw. A rotational row with hi < lo is an outer interval
// wrapping through +-pi.
struct Bounds6 {
  std::array<std::array<double, 2>, 6> rows{};
  double lo(int i) const { return rows[static_cast<std::size_t>(i)][0]; }
  double hi(int i) const { return rows[static_cast<std::size_t>(i)][1]; }
};

// A Task Space Region: sstsr 3.2.0's tsr.TSR, rule for rule (docs/native-design.md, v1.6.0 addendum).
class TSR {
 public:
  TSR(Transform T0_w, Transform Tw_e, Bounds6 Bw);  // throws std::invalid_argument with sstsr's reasons

  const Transform& T0_w() const { return T0_w_; }
  const Transform& Tw_e() const { return Tw_e_; }
  const Bounds6& Bw() const { return Bw_; }
  const Bounds6& continuous_bounds() const { return cont_; }  // sstsr's _Bw_cont

  bool contains(const Transform& T) const;
  double distance(const Transform& T) const;                      // 0 if contained; Berenson 2011 Sec. 4.2 otherwise
  std::pair<double, XyzRpy> distance_bwopt(const Transform& T) const;  // sstsr's distance(): (dist, bwopt)
  std::pair<double, Transform> closest_transform(const Transform& T) const;
  XyzRpy sample_xyzrpy(Rng& rng) const;                            // six unit draws in coordinate order
  Transform sample(Rng& rng) const;
  Transform to_transform(const XyzRpy& v) const;                   // T0_w * xyzrpy_to_trans(v) * Tw_e; validates
  XyzRpy to_xyzrpy(const Transform& T) const;
  double volume() const;                                           // sstsr's _interval_sum

  std::array<bool, 6> is_valid(const XyzRpy& v) const;

 private:
  Transform local(const Transform& T) const;  // inv(T0_w) * T * inv(Tw_e)
  std::array<bool, 3> xyz_within(const std::array<double, 3>& xyz) const;
  std::array<bool, 3> rpy_within(const Rpy& rpy) const;
  std::pair<std::array<bool, 3>, std::optional<Rpy>> rot_within_rpy_bounds(const Transform& local) const;
  std::pair<XyzRpy, XyzRpy> displacement(const Transform& T) const;  // (dx, dw)

  Transform T0_w_, Tw_e_, T0_w_inv_, Tw_e_inv_;
  Bounds6 Bw_, cont_;
};

}  // namespace sscbirrt::tsr
