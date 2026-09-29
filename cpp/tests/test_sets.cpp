#include <algorithm>
#include <memory>
#include <stdexcept>

#include "harness.hpp"
#include "sscbirrt/errors.hpp"
#include "sscbirrt/sets.hpp"

using namespace sscbirrt;

namespace {

std::shared_ptr<FiniteSet> finite(std::vector<Config> m, double tol = 1e-6) { return std::make_shared<FiniteSet>(std::move(m), tol); }

// A closed interval on joint 0 with distance, violation, and projection: a stand-in for a TSR-induced set.
class Interval final : public StateSet, public SetDistance, public SetViolation, public SetProjector {
 public:
  Interval(double lo, double hi) : lo_(lo), hi_(hi) {}
  bool contains(ConfigView q) const override { return q[0] >= lo_ && q[0] <= hi_; }
  double distance(ConfigView q) const override { return q[0] < lo_ ? lo_ - q[0] : (q[0] > hi_ ? q[0] - hi_ : 0.0); }
  double violation(ConfigView q) const override { return distance(q); }
  std::optional<Config> project(ConfigView, ConfigView q) const override {
    Config out = to_config(q);
    out[0] = std::min(std::max(out[0], lo_), hi_);
    return out;
  }
  const SetDistance* distancer() const override { return this; }
  const SetViolation* violator() const override { return this; }
  const SetProjector* projector() const override { return this; }
  std::string describe() const override { return "Interval"; }

 private:
  double lo_, hi_;
};

class Stuck final : public StateSet, public SetViolation, public SetProjector {  // never moves the point
 public:
  bool contains(ConfigView q) const override { return q[0] > 100.0; }
  double violation(ConfigView q) const override { return contains(q) ? 0.0 : 1.0; }
  std::optional<Config> project(ConfigView, ConfigView q) const override { return to_config(q); }
  const SetViolation* violator() const override { return this; }
  const SetProjector* projector() const override { return this; }
  std::string describe() const override { return "Stuck"; }
};

}  // namespace

TEST(finite_set_membership_distance_violation) {
  FiniteSet s({{0.0, 0.0}, {1.0, 0.0}}, 0.1);
  CHECK(s.contains(Config{1.05, 0.0}));
  CHECK(!s.contains(Config{0.5, 0.0}));
  CHECK_NEAR(s.distance(Config{0.5, 0.0}), 0.5, 1e-12);
  CHECK_NEAR(s.violation(Config{0.5, 0.0}), 0.4, 1e-12);
  CHECK_NEAR(s.violation(Config{1.05, 0.0}), 0.0, 1e-12);
  CHECK(s.is_finite() && s.members().size() == 2 && s.members()[1].source == Provenance{1});
  CHECK(s.supports(Capability::Sampler) && !s.supports(Capability::Projector));
  CHECK_THROWS(FiniteSet({}), std::invalid_argument);
  CHECK_THROWS(FiniteSet({{0.0}, {0.0, 1.0}}), std::invalid_argument);
}

TEST(finite_set_samples_one_member_uniformly) {
  FiniteSet s({{0.0}, {1.0}, {2.0}});
  Rng rng(1);
  int seen[3] = {0, 0, 0};
  for (int i = 0; i < 300; ++i) {
    auto smp = s.sample(rng);
    CHECK(smp.size() == 1);
    CHECK(smp[0].source.size() == 1);
    ++seen[smp[0].source[0]];
    CHECK(smp[0].q[0] == static_cast<double>(smp[0].source[0]));
  }
  CHECK(seen[0] > 50 && seen[1] > 50 && seen[2] > 50);
}

TEST(any_of_single_child_delegates_and_prepends_zero) {
  auto u = std::make_shared<AnyOf>(std::vector<SetPtr>{finite({{1.0}})});
  CHECK(u->supports(Capability::Sampler));
  Rng rng(0);
  CHECK(u->sample(rng)[0].source == (Provenance{0, 0}));
  CHECK(u->is_finite() && u->members()[0].source == (Provenance{0, 0}));
}

TEST(any_of_multi_child_needs_weights_to_sample) {
  auto u = std::make_shared<AnyOf>(std::vector<SetPtr>{finite({{1.0}}), finite({{2.0}})});
  CHECK(!u->supports(Capability::Sampler));
  CHECK(u->why_unsupported(Capability::Sampler).find("weights=") != std::string::npos);
  Rng rng(0);
  CHECK_THROWS(u->sample(rng), UnsupportedCapability);
  CHECK(u->supports(Capability::Distance) && u->supports(Capability::Violation));
  CHECK_NEAR(u->distance(Config{1.75}), 0.25, 1e-12);

  auto w = std::make_shared<AnyOf>(std::vector<SetPtr>{finite({{1.0}}), finite({{2.0}})}, std::vector<double>{0.0, 1.0});
  for (int i = 0; i < 20; ++i) CHECK(w->sample(rng)[0].source == (Provenance{1, 0}));
  CHECK_THROWS(AnyOf(std::vector<SetPtr>{finite({{1.0}})}, std::vector<double>{1.0, 1.0}), std::invalid_argument);
  CHECK_THROWS(AnyOf(std::vector<SetPtr>{finite({{1.0}}), finite({{2.0}})}, std::vector<double>{0.0, 0.0}), std::invalid_argument);
  CHECK_THROWS(AnyOf({}), std::invalid_argument);
}

TEST(any_of_projection_picks_nearest_successful_child) {
  auto u = std::make_shared<AnyOf>(std::vector<SetPtr>{std::make_shared<Interval>(0.0, 1.0), std::make_shared<Interval>(5.0, 6.0)});
  CHECK(u->supports(Capability::Projector));
  CHECK_NEAR(u->project(Config{0.0}, Config{4.0}).value()[0], 5.0, 1e-12);
  CHECK_NEAR(u->project(Config{0.0}, Config{2.0}).value()[0], 1.0, 1e-12);
}

TEST(all_of_multi_child_needs_named_strategies) {
  auto a = std::make_shared<Interval>(0.0, 2.0);
  auto b = std::make_shared<Interval>(1.0, 3.0);
  AllOf plain({a, b});
  CHECK(!plain.supports(Capability::Projector) && !plain.supports(Capability::Sampler));
  CHECK(plain.supports(Capability::Violation));
  CHECK_NEAR(plain.violation(Config{-1.0}), 2.0, 1e-12);  // max over children
  CHECK(plain.contains(Config{1.5}) && !plain.contains(Config{0.5}));

  AllOf projected({a, b}, std::make_shared<MostViolatedProjection>());
  CHECK(projected.supports(Capability::Projector));
  CHECK_NEAR(projected.project(Config{0.0}, Config{-1.0}).value()[0], 1.0, 1e-12);

  // MostViolatedProjection requires violation and projection on every child
  CHECK_THROWS(AllOf({a, finite({{0.0}})}, std::make_shared<MostViolatedProjection>()), UnsupportedCapability);
  // RejectionSampling requires the source child to sample
  CHECK_THROWS(AllOf({a, b}, nullptr, std::make_shared<RejectionSampling>(0)), UnsupportedCapability);
  CHECK_THROWS(AllOf({a, b}, nullptr, std::make_shared<RejectionSampling>(5)), std::invalid_argument);
}

TEST(most_violated_projection_gives_up_when_stuck) {
  AllOf stuck({std::make_shared<Interval>(0.0, 1.0), std::make_shared<Stuck>()}, std::make_shared<MostViolatedProjection>(10));
  CHECK(!stuck.project(Config{0.0}, Config{0.5}).has_value());
}

TEST(rejection_sampling_keeps_what_the_others_contain) {
  auto src = finite({{0.5}, {5.0}});
  AllOf inter({src, std::make_shared<Interval>(0.0, 1.0)}, nullptr, std::make_shared<RejectionSampling>(0));
  CHECK(inter.supports(Capability::Sampler));
  Rng rng(0);
  int kept = 0, empty = 0;
  for (int i = 0; i < 100; ++i) {
    auto s = inter.sample(rng);
    if (s.empty()) ++empty;
    else {
      ++kept;
      CHECK(s[0].q[0] == 0.5 && s[0].source == Provenance{0});
    }
  }
  CHECK(kept > 20 && empty > 20);
}

TEST(enumeration_members_and_seeds) {
  auto region = std::make_shared<Interval>(0.0, 10.0);  // not finite
  auto f = finite({{1.0}, {20.0}});
  AnyOf u({f, region}, std::vector<double>{1.0, 1.0});
  CHECK(!u.is_finite() && u.members().empty());
  auto s = u.seeds();
  CHECK(s.size() == 2 && s[0].source == (Provenance{0, 0}) && s[1].source == (Provenance{0, 1}));

  AllOf inter({region, f});
  CHECK(inter.is_finite());
  auto m = inter.members();
  CHECK(m.size() == 1 && m[0].q[0] == 1.0 && m[0].source == Provenance{0});  // AllOf adds no provenance
  CHECK(inter.seeds().size() == 1);

  AllOf nested({std::make_shared<AnyOf>(std::vector<SetPtr>{finite({{1.0}}), finite({{2.0}, {3.0}})}), std::make_shared<Interval>(1.5, 5.0)});
  auto nm = nested.members();
  CHECK(nm.size() == 2 && nm[0].source == (Provenance{1, 0}) && nm[1].source == (Provenance{1, 1}));

  AllOf dedup({std::make_shared<AnyOf>(std::vector<SetPtr>{region, f}, std::vector<double>{1.0, 1.0}),
               std::make_shared<AnyOf>(std::vector<SetPtr>{region, f}, std::vector<double>{1.0, 1.0})});
  CHECK(!dedup.is_finite());
  auto ds = dedup.seeds();
  // Both unions contain both seeds (f has 20), so each child contributes two; duplicates are kept once, earliest child first.
  CHECK(ds.size() == 2 && ds[0].q[0] == 1.0 && ds[0].source == (Provenance{1, 0}) && ds[1].q[0] == 20.0 && ds[1].source == (Provenance{1, 1}));
  EmptySet e;
  CHECK(e.is_finite() && e.members().empty() && !e.contains(Config{0.0}));
}

TEST(root_report_summary_and_only_collisions) {
  RootReport r;
  r.in_collision = 3;
  CHECK(r.only_collisions());
  r.outside_space = 1;
  CHECK(!r.only_collisions());
  CHECK(r.summary() == "1 outside joint space, 3 in collision");
}

HARNESS_MAIN()
