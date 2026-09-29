// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "sscbirrt/types.hpp"

namespace sscbirrt {

// Proposes free-space targets for tree growth. JointSpace is the default (uniform).
// Replacing it trades away probabilistic completeness unless the replacement has
// full support over the space; that is the caller's responsibility.
class SpaceSampler {
 public:
  virtual ~SpaceSampler() = default;
  virtual Config sample(Rng& rng) const = 0;  // a configuration of length dof
};

// Joint-space geometry: limits, metric, direction, interpolation, uniform sampling.
// Topology is the caller's declaration: a bounded joint must have finite limits;
// an angular joint has none and whatever limits were stored for it are ignored.
class JointSpace final : public SpaceSampler {
 public:
  JointSpace(std::vector<double> lower, std::vector<double> upper, std::vector<bool> angular = {});

  int dof() const { return static_cast<int>(lower_.size()); }
  const std::vector<double>& lower() const { return lower_; }
  const std::vector<double>& upper() const { return upper_; }
  const std::vector<bool>& angular() const { return angular_; }
  bool has_angular() const { return has_angular_; }

  bool contains(ConfigView q) const { return !why_invalid(q).has_value(); }
  std::optional<std::string> why_invalid(ConfigView q) const;
  Config direction(ConfigView from, ConfigView to) const;  // short way around angular joints
  double distance(ConfigView a, ConfigView b) const;       // Euclidean norm of direction
  Config interpolate(ConfigView from, ConfigView to, double t) const;
  Config sample(Rng& rng) const override;                  // limits on bounded joints, one turn on angular
  std::vector<Config> unwrap_path(const std::vector<Config>& path) const;

 private:
  std::vector<double> lower_, upper_;
  std::vector<bool> angular_;
  bool has_angular_ = false;
  std::vector<double> sample_lower_, sample_upper_;
};

}  // namespace sscbirrt
