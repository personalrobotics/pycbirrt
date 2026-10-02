// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "sscbirrt/kinematics.hpp"
#include "sscbirrt/sets.hpp"
#include "sscbirrt/space.hpp"
#include "sscbirrt/tsr/tsr.hpp"

namespace sscbirrt::tsr {

// {q : FK(q) in region}: sscbirrt's TSRConfigurationSet (tsr_set.py), rule for rule, for a TSR or a TSR chain.
//   contains:  region distance of FK(q) within tolerance
//   distance:  the region distance of FK(q); violation: max(0, distance - tolerance)
//   sample:    one pose from the region, every IK solution within the space's limits (provenance empty);
//              collision is the planner's validator, so every candidate reaches it
//   project:   move the pose to the region's closest point and solve IK seeded from the current
//              configuration, taking the solution nearest it under the space metric; give up when the
//              distance stops shrinking by progress_tolerance or after max_projection_iters
class TSRConfigurationSet final : public StateSet, public SetSampler, public SetDistance, public SetViolation, public SetProjector {
 public:
  TSRConfigurationSet(Region region, std::shared_ptr<const ForwardKinematics> fk, std::shared_ptr<const IKSolver> ik,
                      std::shared_ptr<const JointSpace> space, double tolerance = 1e-3, int max_projection_iters = 50,
                      double progress_tolerance = 1e-6);

  bool contains(ConfigView q) const override { return distance(q) <= tolerance_; }
  double distance(ConfigView q) const override;
  double violation(ConfigView q) const override;
  std::vector<Sample> sample(Rng& rng) const override;
  std::optional<Config> project(ConfigView q_previous, ConfigView q_proposed) const override;

  const SetSampler* sampler() const override { return this; }
  const SetDistance* distancer() const override { return this; }
  const SetViolation* violator() const override { return this; }
  const SetProjector* projector() const override { return this; }
  std::string describe() const override;

  const Region& region() const { return region_; }
  double tolerance() const { return tolerance_; }

 private:
  bool within_limits(ConfigView q) const;

  Region region_;
  std::shared_ptr<const ForwardKinematics> fk_;
  std::shared_ptr<const IKSolver> ik_;
  std::shared_ptr<const JointSpace> space_;
  double tolerance_;
  int max_projection_iters_;
  double progress_tolerance_;
};

// Mixture weights for an AnyOf of TSR sets, proportional to volume; uniform when every volume is zero
// (sstsr's weights_from_tsrs, as used by the legacy lowering).
std::vector<double> tsr_weights(const std::vector<std::shared_ptr<const TSRConfigurationSet>>& sets);

}  // namespace sscbirrt::tsr
