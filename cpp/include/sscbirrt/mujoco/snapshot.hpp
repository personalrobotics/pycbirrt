// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <string>
#include <vector>

#include "sscbirrt/mujoco/scene.hpp"
#include "sscbirrt/transform.hpp"

namespace sscbirrt::mujoco {

// A grasped object: its body (whose first joint is free) rides on the gripper body at a fixed offset,
// and contacts between its subtree and allowed_bodies are not collisions.
struct Attachment {
  int object_body = -1;
  int gripper_body = -1;
  Transform T_gripper_object = Transform::identity();
  std::vector<int> allowed_bodies;  // resolved in Python from mj_manipulator's gripper-base rule; sorted, unique
};

// Everything a geometric query reads from the live world at one instant, by value. No velocities,
// controls, sensors, warmstarts, forces, or time.
struct Snapshot {
  std::vector<double> qpos;        // nq
  std::vector<double> mocap_pos;   // 3 * nmocap
  std::vector<double> mocap_quat;  // 4 * nmocap, unit within 1e-6
  std::vector<Attachment> attachments;
  std::string sha256;              // filled by validate(): of every field above

  // Checks the snapshot against the scene and fills sha256; throws std::invalid_argument naming the rule.
  void validate(const Scene& scene);
};

}  // namespace sscbirrt::mujoco
