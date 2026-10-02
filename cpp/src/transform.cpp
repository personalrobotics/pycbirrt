// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#include "sscbirrt/transform.hpp"

#include <cmath>

namespace sscbirrt {


Transform Transform::identity() {
  Transform t;
  for (int i = 0; i < 4; ++i) t.at(i, i) = 1.0;
  return t;
}

Transform Transform::from_rows(const std::array<std::array<double, 4>, 4>& rows) {
  Transform t;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) t.at(r, c) = rows[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
  }
  return t;
}

Transform Transform::operator*(const Transform& o) const {
  Transform out;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      double s = 0.0;
      for (int k = 0; k < 4; ++k) s += at(r, k) * o.at(k, c);
      out.at(r, c) = s;
    }
  }
  return out;
}

Transform Transform::inverse_rigid() const {
  Transform out = identity();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) out.at(r, c) = at(c, r);
  }
  for (int r = 0; r < 3; ++r) {
    double s = 0.0;
    for (int k = 0; k < 3; ++k) s += out.at(r, k) * at(k, 3);
    out.at(r, 3) = -s;
  }
  return out;
}

std::optional<std::string> why_not_frame(const Transform& T, const std::string& name) {
  for (double v : T.m) {
    if (!std::isfinite(v)) return name + " must be a finite 4x4 transform";
  }
  const std::array<double, 4> last{0.0, 0.0, 0.0, 1.0};
  for (int c = 0; c < 4; ++c) {
    if (std::fabs(T.at(3, c) - last[static_cast<std::size_t>(c)]) > kFrameAtol) return name + " last row must be [0, 0, 0, 1]";
  }
  for (int r = 0; r < 3; ++r) {  // R R^T = I
    for (int c = 0; c < 3; ++c) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += T.at(r, k) * T.at(c, k);
      if (std::fabs(s - (r == c ? 1.0 : 0.0)) > kFrameAtol) {
        return name + " rotation block is not orthonormal within 1e-06";
      }
    }
  }
  const double det = T.at(0, 0) * (T.at(1, 1) * T.at(2, 2) - T.at(1, 2) * T.at(2, 1)) -
                     T.at(0, 1) * (T.at(1, 0) * T.at(2, 2) - T.at(1, 2) * T.at(2, 0)) +
                     T.at(0, 2) * (T.at(1, 0) * T.at(2, 1) - T.at(1, 1) * T.at(2, 0));
  if (std::fabs(det - 1.0) > kFrameAtol) return name + " rotation block must have determinant +1 (a reflection is not a rotation)";
  return std::nullopt;
}

}  // namespace sscbirrt
