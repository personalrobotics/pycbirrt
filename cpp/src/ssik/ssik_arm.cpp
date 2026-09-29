// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
//
// The only translation unit that sees Eigen and ssik_cpp.
#include "sscbirrt/ssik/ssik_arm.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/solvers/three_parallel.hpp"

namespace sscbirrt::ssik {

namespace {

Eigen::Matrix4d to_eigen(const std::array<double, 16>& m) {
  Eigen::Matrix4d out;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) out(r, c) = m[static_cast<std::size_t>(r * 4 + c)];
  }
  return out;
}

Eigen::Matrix4d to_eigen(const Transform& T) { return to_eigen(T.m); }

Transform from_eigen(const Eigen::Matrix4d& M) {
  Transform T;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) T.at(r, c) = M(r, c);
  }
  return T;
}

}  // namespace

std::optional<Family> family_from_solver_name(const std::string& solver_name) {
  if (solver_name == "ikgeo.three_parallel") return Family::ThreeParallel;
  return std::nullopt;
}

const char* family_name(Family f) {
  switch (f) {
    case Family::ThreeParallel: return "ikgeo.three_parallel";
  }
  return "?";
}

struct SSIKArm::Impl {
  ::ssik::JointConsts<6> consts;
  ::ssik::JointLimits<6> limits;
};

SSIKArm::SSIKArm(ArmSpec spec) : spec_(std::move(spec)), family_(Family::ThreeParallel), impl_(nullptr) {
  const auto fam = family_from_solver_name(spec_.solver_name);
  if (!fam) {
    throw std::invalid_argument("SSIK family " + spec_.solver_name + " is not in the native allowlist (verified: ikgeo.three_parallel)");
  }
  family_ = *fam;
  if (auto why = why_not_frame(spec_.T_base, "T_base")) throw std::invalid_argument(*why);
  if (auto why = why_not_frame(spec_.T_ee, "T_ee")) throw std::invalid_argument(*why);
  auto impl = new Impl();
  for (int i = 0; i < 6; ++i) {
    const auto k = static_cast<std::size_t>(i);
    for (double v : spec_.axis[k]) {
      if (!std::isfinite(v)) { delete impl; throw std::invalid_argument("joint axis must be finite"); }
    }
    impl->consts.axis[k] = Eigen::Vector3d(spec_.axis[k][0], spec_.axis[k][1], spec_.axis[k][2]);
    impl->consts.t_left[k] = to_eigen(spec_.t_left[k]);
    impl->consts.t_right[k] = to_eigen(spec_.t_right[k]);
    impl->consts.type[k] = spec_.revolute[k] ? ::ssik::JointType::Revolute : ::ssik::JointType::Prismatic;
    impl->limits.lo[k] = spec_.lo[k];
    impl->limits.hi[k] = spec_.hi[k];
    impl->limits.present[k] = spec_.present[k];
  }
  impl_ = impl;
}

SSIKArm::~SSIKArm() { delete impl_; }

Transform SSIKArm::fk(ConfigView q) const {
  if (q.size() != 6) throw std::invalid_argument("SSIKArm::fk expects 6 joint values");
  std::array<double, 6> qa{};
  for (std::size_t i = 0; i < 6; ++i) qa[i] = q[i];
  const Eigen::Matrix4d T = to_eigen(spec_.T_base) * ::ssik::fk<6>(impl_->consts, qa) * to_eigen(spec_.T_ee);
  return from_eigen(T);
}

std::vector<Config> SSIKArm::solve(const Transform& pose, ConfigView seed) const {
  const Eigen::Matrix4d target = to_eigen(spec_.T_base).inverse() * to_eigen(pose) * to_eigen(spec_.T_ee).inverse();
  ::ssik::ArtifactParams<6> p;  // defaults: respect_limits, enumerate_windings, no cap, rescue on
  if (!seed.empty()) {
    if (seed.size() != 6) throw std::invalid_argument("SSIKArm::solve seed must have 6 values");
    p.has_seed = true;
    for (std::size_t i = 0; i < 6; ++i) p.q_seed[i] = seed[i];
  }
  std::vector<Config> out;
  switch (family_) {
    case Family::ThreeParallel:
      for (const auto& sol : ::ssik::three_parallel_artifact_solve(impl_->consts, impl_->limits, target, p)) {
        out.emplace_back(sol.q.begin(), sol.q.end());
      }
      break;
  }
  return out;
}

}  // namespace sscbirrt::ssik
