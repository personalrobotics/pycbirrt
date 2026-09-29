#include <numbers>

#include "harness.hpp"
#include "sscbirrt/space.hpp"

using namespace sscbirrt;
constexpr double kPi = std::numbers::pi;
constexpr double kInf = std::numeric_limits<double>::infinity();

TEST(construction_rejects_bad_limits) {
  CHECK_THROWS(JointSpace({0.0, 0.0}, {1.0}), std::invalid_argument);
  CHECK_THROWS(JointSpace({1.0}, {0.0}), std::invalid_argument);
  CHECK_THROWS(JointSpace({0.0, 0.0}, {1.0, 1.0}, {true, true, true}), std::invalid_argument);
}

TEST(bounded_joint_needs_finite_limits_angular_ignores_them) {
  CHECK_THROWS(JointSpace({-1.0, -kInf}, {1.0, kInf}), std::invalid_argument);
  CHECK_THROWS(JointSpace({-1.0}, {kInf}), std::invalid_argument);
  JointSpace s({-kInf, -1.0}, {kInf, 1.0}, {true, false});
  CHECK(s.contains(Config{100.0, 0.0}));
  CHECK(!s.contains(Config{0.0, 2.0}));
  CHECK(s.has_angular());
}

TEST(why_invalid_reports_in_order) {
  JointSpace s({-1.0, -2.0}, {1.0, 2.0});
  CHECK(s.why_invalid(Config{0.0}).value().starts_with("shape"));
  CHECK(s.why_invalid(Config{0.0, kInf}).value().starts_with("non-finite"));
  CHECK(s.why_invalid(Config{0.0, 3.0}).value().starts_with("outside joint limits at joint 1"));
  CHECK(!s.why_invalid(Config{0.0, 0.0}).has_value());
}

TEST(direction_wraps_angular_only) {
  JointSpace s({-kPi, -1.0}, {kPi, 1.0}, {true, false});
  Config d = s.direction(Config{3.0, 0.0}, Config{-3.0, 0.5});
  CHECK_NEAR(d[0], 2 * kPi - 6.0, 1e-12);
  CHECK_NEAR(d[1], 0.5, 1e-12);
  CHECK_NEAR(s.distance(Config{3.0, 0.0}, Config{-3.0, 0.0}), 2 * kPi - 6.0, 1e-12);
  JointSpace linear({-kPi, -1.0}, {kPi, 1.0});
  CHECK_NEAR(linear.distance(Config{3.0, 0.0}, Config{-3.0, 0.0}), 6.0, 1e-12);
}

TEST(interpolate_crosses_the_seam) {
  JointSpace s({-kPi, -1.0}, {kPi, 1.0}, {true, false});
  Config mid = s.interpolate(Config{3.0, 0.0}, Config{-3.0, 0.0}, 0.5);
  CHECK_NEAR(mid[0], 3.0 + (kPi - 3.0), 1e-12);  // halfway around the short way, past +pi
}

TEST(sampling_is_within_limits_seedable_and_one_turn_on_angular) {
  JointSpace s({-1.0, 0.0}, {1.0, 0.0}, {false, true});  // angular joint with degenerate stored limits
  Rng a(3), b(3);
  CHECK(s.sample(a) == s.sample(b));
  Rng rng(0);
  double lo = 10, hi = -10;
  for (int i = 0; i < 500; ++i) {
    Config q = s.sample(rng);
    CHECK(q[0] >= -1.0 && q[0] <= 1.0);
    CHECK(q[1] >= -kPi && q[1] < kPi);
    lo = std::min(lo, q[1]);
    hi = std::max(hi, q[1]);
  }
  CHECK(lo < -2.5 && hi > 2.5);
}

TEST(unwrap_path_keeps_raw_steps_short_on_angular_joints) {
  JointSpace s({-kPi, -1.0}, {kPi, 1.0}, {true, false});
  std::vector<Config> path{{3.0, 0.0}, {-3.1, 0.1}, {-2.9, 0.2}};
  std::vector<Config> out = s.unwrap_path(path);
  CHECK(out[0] == path[0]);
  CHECK_NEAR(out[1][0], 3.0 + (2 * kPi - 6.1), 1e-12);
  CHECK_NEAR(out[2][0], out[1][0] + 0.2, 1e-12);
  CHECK_NEAR(out[2][1], 0.2, 1e-12);
  JointSpace linear({-kPi, -1.0}, {kPi, 1.0});
  CHECK(linear.unwrap_path(path) == path);
}

HARNESS_MAIN()
