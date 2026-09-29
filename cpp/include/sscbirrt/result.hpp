// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sscbirrt/errors.hpp"
#include "sscbirrt/space.hpp"
#include "sscbirrt/types.hpp"

namespace sscbirrt {

enum class Status { Success, Timeout, Aborted, MaxIterations };
const char* status_name(Status s);

struct Node {
  Config q;
  int parent = -1;     // -1 for a root
  Provenance source;   // roots only: which member of the role set
};

// Vector of nodes; nearest is a linear scan under the space metric, ties to the lowest index.
class Tree {
 public:
  Tree() = default;
  explicit Tree(const std::vector<Config>& roots, const std::vector<Provenance>& sources = {});
  int add(Config q, int parent);
  int nearest(const JointSpace& space, ConfigView q) const;
  std::vector<Config> path_to_root(int idx) const;  // root first
  const Provenance& root_source(int idx) const;      // provenance of the root idx descends from
  const std::vector<Node>& nodes() const { return nodes_; }
  int size() const { return static_cast<int>(nodes_.size()); }

 private:
  std::vector<Node> nodes_;
};

struct PlanResult {
  Status status = Status::MaxIterations;
  std::string reason;              // empty on success; prefixes "Timeout", "Aborted", "Max iterations"
  std::vector<Config> path;        // empty unless Success; unwrapped per the JointSpace rule
  Provenance start_source, goal_source;
  int iterations = 0;
  double planning_seconds = 0.0;
  std::pair<int, int> tree_sizes{0, 0};
  RootReport start_roots, goal_roots;
  std::shared_ptr<const Tree> tree_start, tree_goal;  // null unless SolveOptions::keep_trees

  bool success() const { return status == Status::Success; }
};

}  // namespace sscbirrt
