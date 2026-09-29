// The TSR cases of tools/reference_artifact.py (projected_constraint, tsr_goal_union, allof_constraint) run
// natively against the two-link planar arm, checked with the artifact's validation predicates.
#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>

#include "harness.hpp"
#include "sscbirrt/sscbirrt.hpp"

using namespace sscbirrt;
using sscbirrt::tsr::Bounds6;
using sscbirrt::tsr::TSR;
using sscbirrt::tsr::TSRConfigurationSet;
constexpr double kPi = std::numbers::pi;

namespace {

// pycbirrt.testing.PlanarArm / PlanarIK: links 1 and 1, tip pose with identity rotation, both elbow branches.
class PlanarArm final : public ForwardKinematics, public IKSolver {
 public:
  int dof() const override { return 2; }
  Transform fk(ConfigView q) const override {
    Transform T = Transform::identity();
    T.at(0, 3) = std::cos(q[0]) + std::cos(q[0] + q[1]);
    T.at(1, 3) = std::sin(q[0]) + std::sin(q[0] + q[1]);
    return T;
  }
  std::vector<Config> solve(const Transform& pose, ConfigView) const override {
    const double x = pose.at(0, 3), y = pose.at(1, 3);
    const double d = std::sqrt(x * x + y * y);
    if (d > 2.0 || d < 0.0) return {};
    const double c2 = std::clamp((d * d - 2.0) / 2.0, -1.0, 1.0);
    std::vector<Config> out;
    for (double sign : {1.0, -1.0}) {
      const double q2 = sign * std::acos(c2);
      const double q1 = std::atan2(y, x) - std::atan2(std::sin(q2), 1.0 + std::cos(q2));
      out.push_back({q1, q2});
    }
    return out;
  }
};

std::shared_ptr<JointSpace> planar() {
  return std::make_shared<JointSpace>(std::vector<double>{-kPi, -kPi}, std::vector<double>{kPi, kPi});
}
std::shared_ptr<FiniteSet> finite(const JointSpace& space, std::vector<Config> qs) {
  return std::make_shared<FiniteSet>(std::move(qs), 1e-3, [&space](ConfigView a, ConfigView b) { return space.distance(a, b); });
}
Transform frame(double x, double y) {
  Transform T = Transform::identity();
  T.at(0, 3) = x;
  T.at(1, 3) = y;
  return T;
}
const Bounds6 kBox{{{{-0.05, 0.05}, {-0.05, 0.05}, {0, 0}, {0, 0}, {0, 0}, {-kPi, kPi}}}};
const Bounds6 kYBand{{{{-2.0, 2.0}, {-0.6, 0.6}, {0, 0}, {0, 0}, {0, 0}, {-kPi, kPi}}}};

std::shared_ptr<TSRConfigurationSet> tsr_set(const std::shared_ptr<PlanarArm>& arm, const std::shared_ptr<JointSpace>& space,
                                             const Transform& T0_w, const Bounds6& Bw) {
  return std::make_shared<TSRConfigurationSet>(TSR(T0_w, Transform::identity(), Bw), arm, arm, space, 1e-3, 50, 1e-6);
}

PlannerConfig base() {
  PlannerConfig c;
  c.smooth_path = true;
  c.step_size = 0.2;
  c.connection_tolerance = 1e-3;
  c.edge_resolution = 0.05;
  c.timeout_seconds = 30.0;
  return c;
}

PlanningProblem problem(std::shared_ptr<JointSpace> space, SetPtr start, SetPtr goal, SetPtr constraint = nullptr) {
  PlanningProblem p;
  p.space = space;
  p.start = std::move(start);
  p.goal = std::move(goal);
  p.validator = std::make_shared<AcceptAll>();
  p.path_constraint = std::move(constraint);
  return p;
}

bool validate(const PlanningProblem& p, const PlannerConfig& c, const std::vector<Config>& path) {
  const JointSpace& space = *p.space;
  auto admissible = [&p](ConfigView q) { return !Planner::why_inadmissible(p, q).has_value(); };
  bool ok = p.start->contains(path.front()) && p.goal->contains(path.back());
  for (const Config& q : path) ok = ok && space.contains(q) && admissible(q);
  for (std::size_t k = 0; k + 1 < path.size(); ++k) {
    const Config d = space.direction(path[k], path[k + 1]);
    double norm = 0.0, raw = 0.0;
    for (std::size_t i = 0; i < d.size(); ++i) {
      norm += d[i] * d[i];
      raw = std::max(raw, std::fabs(path[k + 1][i] - path[k][i]));
    }
    const int n = std::max(1, static_cast<int>(std::ceil(std::sqrt(norm) / c.resolution())));
    for (int i = 1; i <= n; ++i) {
      Config q(d.size());
      for (std::size_t j = 0; j < d.size(); ++j) q[j] = path[k][j] + (static_cast<double>(i) / n) * d[j];
      ok = ok && admissible(q);
    }
    ok = ok && raw <= c.step_size + 1e-9;
  }
  return ok;
}

SolveOptions seeded(std::uint64_t seed) {
  SolveOptions o;
  o.seed = seed;
  return o;
}

}  // namespace

TEST(lifted_set_capabilities_and_construction) {
  auto arm = std::make_shared<PlanarArm>();
  auto s = planar();
  auto set = tsr_set(arm, s, frame(1.9, 0.3), kBox);
  CHECK(set->supports(Capability::Sampler) && set->supports(Capability::Projector) && set->supports(Capability::Violation));
  CHECK(!set->is_finite());
  Rng rng(0);
  int candidates = 0;
  for (int i = 0; i < 20; ++i) {
    for (const Sample& smp : set->sample(rng)) {
      ++candidates;
      CHECK(smp.source.empty());
      CHECK(set->contains(smp.q));
      CHECK(s->contains(smp.q));
    }
  }
  CHECK(candidates > 20);  // both elbow branches, most of the time
  auto three = std::make_shared<JointSpace>(std::vector<double>{-1, -1, -1}, std::vector<double>{1, 1, 1});
  CHECK_THROWS(TSRConfigurationSet(TSR(frame(0, 0), Transform::identity(), kBox), arm, arm, three), std::invalid_argument);
  CHECK_THROWS(TSRConfigurationSet(TSR(frame(0, 0), Transform::identity(), kBox), arm, arm, s, -1.0), std::invalid_argument);
}

TEST(projection_returns_contained_in_limits_or_nothing) {
  auto arm = std::make_shared<PlanarArm>();
  auto s = planar();
  auto band = tsr_set(arm, s, frame(0, 0), kYBand);
  const Config off{0.0, 1.5};  // tip y = sin(0) + sin(1.5) ~ 1.0 > 0.6
  CHECK(!band->contains(off));
  auto q = band->project(off, off);
  CHECK(q.has_value());
  CHECK(band->contains(*q) && s->contains(*q));
  // Unreachable target: a region the arm cannot reach
  auto far = tsr_set(arm, s, frame(5.0, 0.0), kBox);
  CHECK(!far->project(off, off).has_value());
}

TEST(projected_constraint_case) {
  auto arm = std::make_shared<PlanarArm>();
  auto s = planar();
  Planner planner(base());
  auto p = problem(s, finite(*s, {{0.0, 0.6}}), finite(*s, {{-0.3, 0.9}}), tsr_set(arm, s, frame(0, 0), kYBand));
  PlanResult r = planner.solve(p, seeded(5));
  CHECK(r.success());
  CHECK(validate(p, planner.config(), r.path));
}

TEST(tsr_goal_union_case) {
  auto arm = std::make_shared<PlanarArm>();
  auto s = planar();
  Planner planner(base());
  auto a = tsr_set(arm, s, frame(1.9, 0.3), kBox);
  auto b = tsr_set(arm, s, frame(-1.5, 1.0), kBox);
  const std::vector<double> w = sscbirrt::tsr::tsr_weights({a, b});
  CHECK(w.size() == 2 && w[0] == w[1]);  // same box: same volume
  auto goal = std::make_shared<AnyOf>(std::vector<SetPtr>{a, b}, w);
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), goal);
  PlanResult r = planner.solve(p, seeded(6));
  CHECK(r.success());
  CHECK(validate(p, planner.config(), r.path));
  CHECK(r.goal_source.size() == 1 && (r.goal_source[0] == 0 || r.goal_source[0] == 1));
  CHECK((r.goal_source[0] == 0 ? a : b)->contains(r.path.back()));
  CHECK(r.goal_roots.draws > 0);
}

TEST(allof_constraint_case) {
  auto arm = std::make_shared<PlanarArm>();
  auto s = planar();
  Planner planner(base());
  auto x_band = tsr_set(arm, s, frame(1.5, 0.0), Bounds6{{{{-0.5, 0.5}, {-2, 2}, {0, 0}, {0, 0}, {0, 0}, {-kPi, kPi}}}});
  auto y_band = tsr_set(arm, s, frame(0, 0), kYBand);
  auto both = std::make_shared<AllOf>(std::vector<SetPtr>{x_band, y_band}, std::make_shared<MostViolatedProjection>());
  CHECK(both->supports(Capability::Projector));
  auto p = problem(s, finite(*s, {{0.0, 0.6}}), finite(*s, {{-0.3, 0.9}}), both);
  PlanResult r = planner.solve(p, seeded(8));
  CHECK(r.success());
  CHECK(validate(p, planner.config(), r.path));
}

HARNESS_MAIN()
