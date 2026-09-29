// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <memory>

#include "sscbirrt/motion.hpp"
#include "sscbirrt/sets.hpp"
#include "sscbirrt/space.hpp"
#include "sscbirrt/validity.hpp"

namespace sscbirrt {

// Roles, not meanings: a set does not know which role it plays. Copying the problem shares
// the components. The solver borrows it for the duration of solve.
struct PlanningProblem {
  std::shared_ptr<const JointSpace> space;
  std::shared_ptr<const StateSet> start;  // finite, sampleable, or both
  std::shared_ptr<const StateSet> goal;   // same requirement
  std::shared_ptr<const StateValidator> validator;
  std::shared_ptr<const StateSet> path_constraint;        // may be null
  std::shared_ptr<const MotionValidator> motion_validator;  // null means the default discretized check
  std::shared_ptr<const SpaceSampler> sampler;              // null means *space
};

}  // namespace sscbirrt
