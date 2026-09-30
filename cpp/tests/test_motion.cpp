#include <algorithm>
#include <memory>
#include <stdexcept>

#include "harness.hpp"
#include "sscbirrt/motion.hpp"
#include "sscbirrt/validity.hpp"

using namespace sscbirrt;

namespace {
auto space2() { return std::make_shared<JointSpace>(std::vector<double>{-3.0, -3.0}, std::vector<double>{3.0, 3.0}); }
}  // namespace

TEST(box_obstacles_are_open_and_may_be_unbounded) {
  const double inf = std::numeric_limits<double>::infinity();
  JointBoxObstacles wall({{{0.45, -inf}, {0.55, inf}}});  // sscbirrt.testing.Wall(axis=0, lo=0.45, hi=0.55)
  CHECK(!wall.is_valid(Config{0.5, 2.0}));
  CHECK(wall.is_valid(Config{0.45, 0.0}));  // boundary is outside (open box)
  CHECK(wall.is_valid(Config{0.6, 0.0}));
  JointBoxObstacles slab({{{-0.2, -1.0}, {0.2, 1.0}}});  // Wall(axis=0, -0.2, 0.2, extent=1.0)
  CHECK(!slab.is_valid(Config{0.0, 0.5}));
  CHECK(slab.is_valid(Config{0.0, 1.5}));
  CHECK_THROWS(JointBoxObstacles({{{0.0}, {1.0, 2.0}}}), std::invalid_argument);
}

TEST(discrete_validator_samples_at_resolution_and_ends_exactly) {
  auto s = space2();
  DiscreteMotionValidator v(s, [](ConfigView) { return true; }, 0.3);
  LocalMotion m = v.validate(Config{0.0, 0.0}, Config{1.0, 0.0});
  CHECK(m.reached);
  CHECK(m.configs.size() == 4);  // ceil(1.0 / 0.3) = 4
  CHECK(m.configs.back() == (Config{1.0, 0.0}));
  CHECK_NEAR(m.configs[0][0], 0.25, 1e-12);
  CHECK(v.validate(Config{0.0, 0.0}, Config{0.0, 0.0}).reached);
  CHECK(v.validate(Config{0.0, 0.0}, Config{0.0, 0.0}).configs.empty());
  CHECK(!v.validate(Config{0.0, 0.0}, Config{9.0, 0.0}).reached);  // target outside the space
}

TEST(discrete_validator_keeps_the_admissible_prefix) {
  auto s = space2();
  DiscreteMotionValidator v(s, [](ConfigView q) { return q[0] < 0.6; }, 0.25);
  LocalMotion m = v.validate(Config{0.0, 0.0}, Config{1.0, 0.0});
  CHECK(!m.reached);
  CHECK(m.configs.size() == 2);  // 0.25, 0.5 admissible; 0.75 is not
  CHECK_THROWS(DiscreteMotionValidator(s, [](ConfigView) { return true; }, 0.0), std::invalid_argument);
}

TEST(restricted_validator_rejects_whole_motion) {
  auto s = space2();
  auto base = std::make_shared<DiscreteMotionValidator>(s, [](ConfigView) { return true; }, 0.5);
  RestrictedMotionValidator r(base, [](ConfigView a, ConfigView b) { return std::fabs(b[0] - a[0]) < 0.5; });
  CHECK(r.validate(Config{0.0, 0.0}, Config{0.4, 0.0}).reached);
  LocalMotion m = r.validate(Config{0.0, 0.0}, Config{1.0, 0.0});
  CHECK(!m.reached && m.configs.empty());
}

HARNESS_MAIN()
