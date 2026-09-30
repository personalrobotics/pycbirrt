// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#include "sscbirrt/mujoco/snapshot.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sscbirrt/detail/sha256.hpp"

namespace sscbirrt::mujoco {

namespace {
void hash_doubles(detail::Sha256& h, const std::vector<double>& v) { h.update(v.data(), v.size() * sizeof(double)); }
void hash_int(detail::Sha256& h, int v) { h.update(&v, sizeof v); }
}  // namespace

void Snapshot::validate(const Scene& scene) {
  if (static_cast<int>(qpos.size()) != scene.nq()) {
    throw std::invalid_argument("snapshot qpos has " + std::to_string(qpos.size()) + " entries; the scene has nq = " + std::to_string(scene.nq()));
  }
  for (double v : qpos) {
    if (!std::isfinite(v)) throw std::invalid_argument("snapshot qpos has a non-finite entry");
  }
  if (static_cast<int>(mocap_pos.size()) != 3 * scene.nmocap() || static_cast<int>(mocap_quat.size()) != 4 * scene.nmocap()) {
    throw std::invalid_argument("snapshot mocap arrays must have 3*nmocap and 4*nmocap entries (nmocap = " + std::to_string(scene.nmocap()) + ")");
  }
  for (double v : mocap_pos) {
    if (!std::isfinite(v)) throw std::invalid_argument("snapshot mocap_pos has a non-finite entry");
  }
  for (int i = 0; i < scene.nmocap(); ++i) {
    double n2 = 0.0;
    for (int k = 0; k < 4; ++k) {
      const double v = mocap_quat[static_cast<std::size_t>(4 * i + k)];
      if (!std::isfinite(v)) throw std::invalid_argument("snapshot mocap_quat has a non-finite entry");
      n2 += v * v;
    }
    if (std::fabs(std::sqrt(n2) - 1.0) > 1e-6) throw std::invalid_argument("snapshot mocap_quat " + std::to_string(i) + " is not a unit quaternion");
  }
  for (std::size_t i = 0; i < attachments.size(); ++i) {
    Attachment& a = attachments[i];
    const std::string where = "attachment " + std::to_string(i);
    if (a.object_body < 0 || a.object_body >= scene.nbody()) throw std::invalid_argument(where + ": object body id out of range");
    if (!scene.body_has_free_joint(a.object_body)) {
      throw std::invalid_argument(where + ": object body '" + scene.body_name(a.object_body) + "' has no free joint as its first joint");
    }
    if (a.gripper_body < 0 || a.gripper_body >= scene.nbody()) throw std::invalid_argument(where + ": gripper body id out of range");
    if (!scene.is_arm_body(a.gripper_body)) {
      throw std::invalid_argument(where + ": gripper body '" + scene.body_name(a.gripper_body) + "' is not part of the arm");
    }
    if (auto why = why_not_frame(a.T_gripper_object, where + ": T_gripper_object")) throw std::invalid_argument(*why);
    for (int b : a.allowed_bodies) {
      if (b < 0 || b >= scene.nbody()) throw std::invalid_argument(where + ": allowed body id out of range");
    }
    std::sort(a.allowed_bodies.begin(), a.allowed_bodies.end());
    a.allowed_bodies.erase(std::unique(a.allowed_bodies.begin(), a.allowed_bodies.end()), a.allowed_bodies.end());
  }
  // Order of attachments never changes a decision; hash them in a canonical order.
  std::sort(attachments.begin(), attachments.end(), [](const Attachment& x, const Attachment& y) { return x.object_body < y.object_body; });
  for (std::size_t i = 1; i < attachments.size(); ++i) {
    if (attachments[i].object_body == attachments[i - 1].object_body) {
      throw std::invalid_argument("object body '" + scene.body_name(attachments[i].object_body) + "' is attached twice");
    }
  }

  detail::Sha256 h;
  hash_doubles(h, qpos);
  hash_doubles(h, mocap_pos);
  hash_doubles(h, mocap_quat);
  for (const Attachment& a : attachments) {
    hash_int(h, a.object_body);
    hash_int(h, a.gripper_body);
    h.update(a.T_gripper_object.m.data(), a.T_gripper_object.m.size() * sizeof(double));
    for (int b : a.allowed_bodies) hash_int(h, b);
  }
  const auto d = h.digest();
  static const char* digits = "0123456789abcdef";
  sha256.clear();
  for (std::uint8_t b : d) {
    sha256.push_back(digits[b >> 4]);
    sha256.push_back(digits[b & 15]);
  }
}

}  // namespace sscbirrt::mujoco
