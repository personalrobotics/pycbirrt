// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace sscbirrt {

// Per-role account of root collection; carried by the result and by NoRoots.
struct RootReport {
  int explicit_candidates = 0;  // from seeds()
  int explicit_rejected = 0;    // of those, how many were inadmissible
  int draws = 0;                // sampling draws made
  int draws_empty = 0;          // draws that produced no candidate ("IK unreachable" in Python's summary)
  int outside_space = 0;        // rejections by reason, explicit and sampled together
  int in_collision = 0;
  int constraint_violated = 0;
  int roots = 0;
  std::vector<std::string> details;  // one line per explicit rejection, as Python logs them

  int rejections() const { return draws_empty + outside_space + in_collision + constraint_violated; }
  // Every rejection was the validator's: Python raises All...InCollision rather than All...Invalid.
  bool only_collisions() const { return in_collision > 0 && in_collision == rejections(); }
  std::string summary() const;  // Python's sampling summary, e.g. "2 IK unreachable, 3 in collision"
};

// A set asked for a capability it lacks, or a start/goal set that is neither finite nor sampleable.
class UnsupportedCapability : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// A component violated an invariant the planner relies on (LocalMotion contract, wrong-length configuration).
class ContractError : public std::logic_error {
 public:
  using std::logic_error::logic_error;
};

// No admissible root for a role. Python raises All{Start,Goal}Configurations{Invalid,InCollision}.
class NoRoots : public std::runtime_error {
 public:
  NoRoots(std::string role_, RootReport report_, const std::string& message)
      : std::runtime_error(message), role(std::move(role_)), report(std::move(report_)) {}
  std::string role;  // "start" or "goal"
  RootReport report;
};

}  // namespace sscbirrt
