// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#include "sscbirrt/space.hpp"

#include <cmath>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace sscbirrt {

namespace {
constexpr double kPi = std::numbers::pi;

std::string fmt(double v) {
  std::ostringstream os;
  os.precision(4);
  os << v;
  return os.str();
}
}  // namespace

JointSpace::JointSpace(std::vector<double> lower, std::vector<double> upper, std::vector<bool> angular)
    : lower_(std::move(lower)), upper_(std::move(upper)), angular_(std::move(angular)) {
  if (lower_.size() != upper_.size() || lower_.empty()) {
    throw std::invalid_argument("lower and upper must be nonempty arrays of the same length");
  }
  for (std::size_t i = 0; i < lower_.size(); ++i) {
    if (lower_[i] > upper_[i]) throw std::invalid_argument("lower limits must not exceed upper limits");
  }
  if (!angular_.empty() && angular_.size() != lower_.size()) {
    throw std::invalid_argument("angular_joints length (" + std::to_string(angular_.size()) +
                                ") must match robot DOF (" + std::to_string(lower_.size()) + ")");
  }
  if (angular_.empty()) angular_.assign(lower_.size(), false);
  for (bool a : angular_) has_angular_ = has_angular_ || a;

  // Topology is the caller's declaration: bounded joints need finite limits; angular
  // joints ignore theirs and sample over one full turn.
  for (std::size_t i = 0; i < lower_.size(); ++i) {
    if (!angular_[i] && !(std::isfinite(lower_[i]) && std::isfinite(upper_[i]))) {
      throw std::invalid_argument("joint " + std::to_string(i) + " has non-finite limits [" + fmt(lower_[i]) + ", " +
                                  fmt(upper_[i]) + "]; give finite planning limits or mark it angular (angular_joints)");
    }
  }
  sample_lower_ = lower_;
  sample_upper_ = upper_;
  for (std::size_t i = 0; i < lower_.size(); ++i) {
    if (angular_[i]) {
      sample_lower_[i] = -kPi;
      sample_upper_[i] = kPi;
    }
  }
}

std::optional<std::string> JointSpace::why_invalid(ConfigView q) const {
  if (q.size() != lower_.size()) {
    return "shape (" + std::to_string(q.size()) + ",) != (" + std::to_string(lower_.size()) + ",)";
  }
  for (double v : q) {
    if (!std::isfinite(v)) return std::string("non-finite entries");
  }
  std::string bad;
  for (std::size_t i = 0; i < q.size(); ++i) {
    if (angular_[i]) continue;
    if (q[i] < lower_[i] || q[i] > upper_[i]) {
      if (!bad.empty()) bad += ", ";
      bad += "joint " + std::to_string(i) + ": " + fmt(q[i]) + " not in [" + fmt(lower_[i]) + ", " + fmt(upper_[i]) + "]";
    }
  }
  if (!bad.empty()) return "outside joint limits at " + bad;
  return std::nullopt;
}

Config JointSpace::direction(ConfigView from, ConfigView to) const {
  Config d(from.size());
  for (std::size_t i = 0; i < from.size(); ++i) {
    d[i] = to[i] - from[i];
    if (angular_[i]) d[i] = std::atan2(std::sin(d[i]), std::cos(d[i]));
  }
  return d;
}

double JointSpace::distance(ConfigView a, ConfigView b) const {
  double s = 0.0;
  for (double d : direction(a, b)) s += d * d;
  return std::sqrt(s);
}

Config JointSpace::interpolate(ConfigView from, ConfigView to, double t) const {
  Config d = direction(from, to);
  for (std::size_t i = 0; i < d.size(); ++i) d[i] = from[i] + t * d[i];
  return d;
}

Config JointSpace::sample(Rng& rng) const {
  Config q(lower_.size());
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = sample_lower_[i] + unit(rng) * (sample_upper_[i] - sample_lower_[i]);
  }
  return q;
}

std::vector<Config> JointSpace::unwrap_path(const std::vector<Config>& path) const {
  if (!has_angular_ || path.size() < 2) return path;
  std::vector<Config> out;
  out.reserve(path.size());
  out.push_back(path.front());
  for (std::size_t k = 1; k < path.size(); ++k) {
    const Config d = direction(out.back(), path[k]);
    Config next(d.size());
    for (std::size_t i = 0; i < d.size(); ++i) next[i] = angular_[i] ? out.back()[i] + d[i] : path[k][i];
    out.push_back(std::move(next));
  }
  return out;
}

}  // namespace sscbirrt
