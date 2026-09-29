// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <memory>

#include "sscbirrt/cancel.hpp"
#include "sscbirrt/config.hpp"
#include "sscbirrt/motion.hpp"
#include "sscbirrt/problem.hpp"
#include "sscbirrt/result.hpp"

namespace sscbirrt {

// CBiRRT over sets. solve() is const and reentrant: every solve owns its own RNG and trees.
class Planner {
 public:
  explicit Planner(PlannerConfig config = {});
  const PlannerConfig& config() const { return config_; }

  // Throws std::invalid_argument (malformed input), UnsupportedCapability (a start or goal set that is
  // neither finite nor sampleable), NoRoots (no admissible root for a role), ContractError (a component
  // violated an invariant). Search outcomes are statuses.
  PlanResult solve(const PlanningProblem& problem, const SolveOptions& options = {}) const;

  // The validator used when problem.motion_validator is null; wrap it in RestrictedMotionValidator
  // to add a restriction while keeping the discrete checks.
  std::shared_ptr<DiscreteMotionValidator> default_motion_validator(const PlanningProblem& problem) const;

  // Whether q may appear on a path, and if not, why: space, then validator, then path constraint.
  static std::optional<std::string> why_inadmissible(const PlanningProblem& problem, ConfigView q);

 private:
  PlannerConfig config_;
};

}  // namespace sscbirrt
