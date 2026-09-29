// The minimal standalone consumer from docs/native-design.md: a two-joint space, one fixed start,
// two fixed goals, and a wall between them. Exits 0 on a validated path.
#include <sscbirrt/sscbirrt.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>

int main() {
  using namespace sscbirrt;
  const double pi = 3.14159265358979323846;
  const double inf = std::numeric_limits<double>::infinity();

  auto space = std::make_shared<JointSpace>(std::vector<double>{-pi, -pi}, std::vector<double>{pi, pi});
  auto start = std::make_shared<FiniteSet>(std::vector<Config>{{-2.0, 0.5}});
  auto goal = std::make_shared<FiniteSet>(std::vector<Config>{{2.0, -0.5}, {2.0, 0.5}});
  auto walls = std::make_shared<JointBoxObstacles>(std::vector<JointBoxObstacles::Box>{
      {{-0.2, -inf}, {0.2, 1.0}}});  // a slab at q0 in (-0.2, 0.2) for q1 below 1.0

  PlanningProblem problem;
  problem.space = space;
  problem.start = start;
  problem.goal = goal;
  problem.validator = walls;

  PlannerConfig config;
  config.step_size = 0.2;
  config.timeout_seconds = 5.0;

  SolveOptions options;
  options.seed = 0;
  PlanResult r = Planner(config).solve(problem, options);
  if (!r.success()) {
    std::printf("no path: %s\n", r.reason.c_str());
    return 1;
  }
  // Independent check: every waypoint admissible, every raw step within one step size.
  for (std::size_t k = 0; k < r.path.size(); ++k) {
    if (Planner::why_inadmissible(problem, r.path[k])) return 2;
    if (k > 0 && space->distance(r.path[k - 1], r.path[k]) > config.step_size + 1e-9) return 3;
  }
  std::printf("%zu waypoints in %d iterations; goal member %d\n", r.path.size(), r.iterations, r.goal_source.back());
  return 0;
}
