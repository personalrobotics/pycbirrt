# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Example: Euclidean vs kinetic-energy metric in joint space.

An RRT measures distance in joint space, and the default -- a plain Euclidean
norm on joint angles -- assumes every joint is equally expensive to move. On a
serial arm that is false: the shoulder swings the whole limb, the wrist moves
only the hand. A UR5e's mass matrix has eigenvalues spanning a factor of ~37,
so "0.1 radians" costs very different amounts of work depending on the joint.

The kinetic-energy metric measures a displacement by the work it takes,

    d(q, q + dq) = sqrt( dq^T M(q) dq )

which is the Riemannian metric induced by kinetic energy T = 1/2 qdot^T M qdot.
Nearest-neighbour queries and step sizes then reflect what the arm actually has
to do, so the tree grows preferentially along cheap directions.

This script plans the same query under both metrics and reports, for each:

  * the **kinetic work** along the path, integral sqrt(dq^T M dq), which is the
    quantity the kinetic metric is trying to minimise;
  * the **joint travel**, sum |dq|, which the Euclidean metric minimises;
  * how much of the motion each joint contributes, showing whether the planner
    leaned on the heavy proximal joints or the light distal ones.

``M(q)`` varies along a path, so *where* it is sampled matters. The metric
defaults to the **midpoint** rule, which is second-order accurate: against a
finely subdivided arc length on a UR5e it is ~25x closer than evaluating at the
segment start (0.06% vs 1.4% at a typical 0.2 rad step).

``--geodesic-extension`` is the change that actually matters. It extends the
tree along the metric's **natural gradient** of the squared-distance potential
instead of the straight line -- Algorithm 1 of Kyaw & Kelly, *Geometry-Aware
Sampling-Based Motion Planning on Riemannian Manifolds* (arXiv:2602.00992).
Under an anisotropic metric the two differ: on a UR5e the natural-gradient
direction sits at ~0.91 cosine to the straight line, and it is the straight
line that is wrong -- steepest descent turns away from the heavy directions.

``--metric-sampling`` additionally draws random samples proportional to the
metric's volume element ``sqrt(det M(q))`` rather than uniformly in joint
coordinates.

**What to expect**, measured over ~60 paired queries each:

  ==========================  ==============  ==============
  variant                     free space      with obstacle
  ==========================  ==============  ==============
  kinetic metric only         -0.3% (noise)   -7.7% (noise)
  kinetic + geodesic ext.     -10.3%          **-42.5%**
  ==========================  ==============  ==============

The metric *alone* barely helps, and the reason is instructive: CBiRRT returns
the first path that connects and never compares alternatives, and its two trees
meet in a median of **1 iteration**, so a nearest-neighbour metric has almost no
tree growth to influence. Changing how each *edge* is traced is what pays off --
with geodesic extension the obstacle result reaches t = -2.5 and wins 55/68
queries, where the metric alone stays inside noise.

Metric-aware sampling measurably shifts the sample distribution (~+15% mean
volume element) but does not improve path work here: the volume element varies
only ~2.3x across this workspace. It is off by default.

The report prints a paired mean +/- standard error so these can be read
honestly rather than inferred from a difference of means; RRT run-to-run spread
is large enough that a difference of means alone is easy to over-read.

Run with:
    python examples/metric_comparison.py
    python examples/metric_comparison.py --queries 12 --seed 7
    python examples/metric_comparison.py --viz     # side-by-side paths in viser
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _circle_array import write_circle_array  # noqa: E402


class NoCollision:
    """Free space."""

    def is_valid(self, q) -> bool:  # noqa: D102
        return True


class JointSlab:
    """Forbids a slab in joint space, forcing a detour around it.

    In *free* space the straight line between two configurations is already
    very nearly work-optimal (a search over random detours improves on it by
    well under 1%), so no metric can do much there. The metric earns its keep
    when something blocks the direct route and the planner has to choose which
    way around -- through the heavy proximal joints, or the light distal ones.
    """

    def __init__(self, joints=(1, 2), half_width: float = 0.35):
        self.joints = tuple(joints)
        self.half_width = float(half_width)

    def is_valid(self, q) -> bool:
        return not all(abs(q[j]) < self.half_width for j in self.joints)


def build_arm(out_dir: Path | None = None):
    """A single UR5e (the array composer with one arm), plus its model."""
    import gafro as ga

    from pycbirrt.backends.gafro import GafroRobotModel

    # Default to a temp directory: writing the composed arm next to the source
    # leaves a generated file in the repo.
    directory = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="metric_arm_"))
    path = write_circle_array(1, directory / "metric_comparison_arm.yaml")
    system = ga.SystemSerialization.load(str(path))
    return GafroRobotModel(system, chain_name="arm0/ur5e_ee"), system


def kinetic_work(metric, path) -> float:
    """Integral of sqrt(dq^T M dq) along a path -- the work the motion costs."""
    return float(sum(metric.norm(path[i], path[i + 1] - path[i]) for i in range(len(path) - 1)))


def joint_travel(path) -> float:
    """Total |dq| summed over the path -- what the Euclidean metric minimises."""
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def per_joint_travel(path) -> np.ndarray:
    """Absolute travel of each joint along the path."""
    return np.abs(np.diff(np.asarray(path), axis=0)).sum(axis=0)


def plan_once(model, solver, goal_tsr, start, metric, step_size, seed, collision=None,
              metric_sampling: bool = False, geodesic_extension: bool = False):
    """Plan one query under one metric; returns (path, seconds) or (None, seconds)."""
    from pycbirrt import CBiRRT, CBiRRTConfig
    from pycbirrt.exceptions import PlanningError

    collision = collision if collision is not None else NoCollision()
    config = CBiRRTConfig(max_iterations=4000, step_size=step_size, goal_bias=0.3,
                          tsr_samples=30, angular_joints=(True,) * model.dof,
                          metric=metric, metric_sampling=metric_sampling,
                          geodesic_extension=geodesic_extension, smooth_path=True)
    planner = CBiRRT(model, solver, collision, config)
    started = time.time()
    try:
        result = planner.plan(start=start, goal_tsrs=[goal_tsr], seed=seed,
                              return_details=True)
    except PlanningError:
        # An unreachable start/goal is a property of the query, not the metric;
        # such queries are dropped so the comparison stays paired.
        return None, time.time() - started
    elapsed = time.time() - started
    return (result.path if result.success else None), elapsed


def goal_tsr_for(model, q_goal, tolerance: float = 0.05):
    """A small TSR centred on the pose that ``q_goal`` reaches."""
    from tsr import TSR

    T0_w = model.forward_kinematics(q_goal).to_transformation_matrix()
    Bw = np.array([[-tolerance, tolerance]] * 3 + [[-0.03, 0.03]] * 3)
    return TSR(T0_w=T0_w, Tw_e=np.eye(4), Bw=Bw)


def run_comparison(queries: int, seed: int, step_size: float, out_dir: Path | None = None,
                   collision=None, model=None, kinetic=None, solver=None,
                   metric_sampling: bool = False, geodesic_extension: bool = False):
    """Plan several random queries under both metrics and tabulate the result."""
    from pycbirrt.backends.gafro import GafroIKSolver
    from pycbirrt.metrics import metric_for_model

    collision = collision if collision is not None else NoCollision()
    if model is None:
        model, _system = build_arm(out_dir)
    if kinetic is None:
        kinetic = metric_for_model(model)
    if solver is None:
        solver = GafroIKSolver(model.manipulator, model.joint_limits, max_iterations=200,
                               tolerance=1e-5, base_configuration=model.base_configuration,
                               collision_checker=collision)

    rng = np.random.default_rng(seed)
    # The UR5e's limits are +/-2pi, so sampling them broadly gives folded-up
    # configurations whose poses IK cannot recover. Draw modest joint angles
    # instead: the comparison is about the metric, not about reachability.
    spread = 0.9

    rows = []
    attempted = 0
    while len(rows) < queries and attempted < queries * 4:
        index = attempted
        attempted += 1
        start = rng.uniform(-spread, spread, model.dof)
        goal_q = rng.uniform(-spread, spread, model.dof)
        if not (collision.is_valid(start) and collision.is_valid(goal_q)):
            continue
        goal = goal_tsr_for(model, goal_q)

        result = {}
        for name, metric in (("euclidean", None), ("kinetic", kinetic)):
            path, elapsed = plan_once(model, solver, goal, start, metric, step_size,
                                      seed=1000 + index, collision=collision,
                                      metric_sampling=metric_sampling and metric is not None,
                                      geodesic_extension=geodesic_extension and metric is not None)
            result[name] = (path, elapsed)

        if result["euclidean"][0] is None or result["kinetic"][0] is None:
            continue

        row = {"query": index}
        for name, (path, elapsed) in result.items():
            row[name] = {
                "work": kinetic_work(kinetic, path),
                "travel": joint_travel(path),
                "waypoints": len(path),
                "seconds": elapsed,
                "per_joint": per_joint_travel(path),
                "path": path,
            }
        rows.append(row)

    return model, kinetic, rows


def report(model, kinetic, rows) -> None:
    """Print the per-query table and the aggregate the comparison is about."""
    if not rows:
        print("no comparable queries")
        return

    print()
    print("per query (work = integral sqrt(dq' M dq); travel = sum |dq|)")
    print(f"{'query':>5}  {'euclid work':>11} {'kinetic work':>12} {'work diff':>10}"
          f"  {'euclid travel':>13} {'kinetic travel':>14}")
    for row in rows:
        euclid, kin = row["euclidean"], row["kinetic"]
        change = 100.0 * (kin["work"] - euclid["work"]) / euclid["work"]
        print(f"{row['query']:>5}  {euclid['work']:>11.3f} {kin['work']:>12.3f} "
              f"{change:>9.1f}%  {euclid['travel']:>13.3f} {kin['travel']:>14.3f}")

    work_e = np.array([r["euclidean"]["work"] for r in rows])
    work_k = np.array([r["kinetic"]["work"] for r in rows])
    travel_e = np.array([r["euclidean"]["travel"] for r in rows])
    travel_k = np.array([r["kinetic"]["travel"] for r in rows])

    # Paired statistics: RRT run-to-run spread is large, so the per-query
    # difference matters more than the difference of the means.
    delta = work_k - work_e
    stderr = delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else float("nan")

    print()
    print(f"over {len(rows)} queries:")
    print(f"  kinetic work   {work_e.mean():.3f} -> {work_k.mean():.3f} "
          f"({100.0 * (work_k.mean() - work_e.mean()) / work_e.mean():+.1f}%)   "
          f"kinetic wins {int((work_k < work_e).sum())}/{len(rows)}")
    print(f"    paired difference {delta.mean():+.3f} +/- {stderr:.3f} (1 s.e.)"
          f"{'  -- within noise' if abs(delta.mean()) < 2 * stderr else ''}")
    print(f"  joint travel   {travel_e.mean():.3f} -> {travel_k.mean():.3f} "
          f"({100.0 * (travel_k.mean() - travel_e.mean()) / travel_e.mean():+.1f}%)   "
          f"euclidean wins {int((travel_e < travel_k).sum())}/{len(rows)}")

    joints_e = np.mean([r["euclidean"]["per_joint"] for r in rows], axis=0)
    joints_k = np.mean([r["kinetic"]["per_joint"] for r in rows], axis=0)
    print()
    print("mean travel per joint (joint 0 = shoulder, heaviest):")
    print(f"  {'joint':>5} {'euclidean':>10} {'kinetic':>9} {'change':>8}   cost of 0.1 rad")
    reference = np.zeros(model.dof)
    for j in range(model.dof):
        step = np.zeros(model.dof)
        step[j] = 0.1
        change = 100.0 * (joints_k[j] - joints_e[j]) / joints_e[j] if joints_e[j] > 1e-9 else 0.0
        print(f"  {j:>5} {joints_e[j]:>10.3f} {joints_k[j]:>9.3f} {change:>7.1f}%   "
              f"{kinetic.norm(reference, step):.4f}")


def visualize(model, rows, port: int) -> None:
    """Overlay the two paths for the first query, with a playback scrubber."""
    import gafro as ga
    from circle_array_tsr import _patch_visualizer_joint_limits

    if not rows:
        print("nothing to visualize")
        return
    _patch_visualizer_joint_limits()

    viz = ga.Visualizer(port=port)
    robot_viz = viz.add_robot(model.system, joint_sliders=False)

    paths = {"euclidean": rows[0]["euclidean"], "kinetic": rows[0]["kinetic"]}
    colors = {"euclidean": (80, 140, 255), "kinetic": (255, 140, 40)}
    for name, entry in paths.items():
        points = [model.forward_kinematics(q).get_translator() for q in entry["path"]]
        viz.add_spline([[p.x(), p.y(), p.z()] for p in points],
                       name=f"/{name}", color=colors[name])
        viz.add_label(f"{name}: work {entry['work']:.3f}, travel {entry['travel']:.3f}",
                      position=(0.0, 0.0, 0.95 if name == "euclidean" else 0.85),
                      name=f"/label_{name}")

    longest = max(paths.values(), key=lambda e: len(e["path"]))["path"]
    viz.add_playback(list(longest),
                     lambda _i, q: robot_viz.update(model.to_system_configuration(q)))
    viz.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=int, default=8, help="random queries to plan (default: 8)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the queries (default: 0)")
    parser.add_argument("--step-size", type=float, default=0.2, help="planner step size")
    parser.add_argument("--viz", action="store_true", help="show the first query's two paths")
    parser.add_argument("--port", type=int, default=8080, help="viser port")
    parser.add_argument("--out-dir", default=None, help="where to write the arm description")
    parser.add_argument("--free-only", action="store_true",
                        help="only the free-space scenario (skip the obstacle)")
    parser.add_argument("--metric-sampling", action="store_true",
                        help="also weight random samples by sqrt(det M)")
    parser.add_argument("--geodesic-extension", action="store_true",
                        help="extend along the natural gradient (Kyaw & Kelly Alg. 1)")
    parser.add_argument("--evaluate-at", default="midpoint",
                        choices=("midpoint", "average", "start"),
                        help="where the metric samples M (default: midpoint)")
    args = parser.parse_args()

    from pycbirrt.backends.gafro import GafroIKSolver
    from pycbirrt.metrics import metric_for_model

    print("Euclidean vs kinetic-energy metric on a UR5e")
    model, _system = build_arm(args.out_dir)
    kinetic = metric_for_model(model, evaluate_at=args.evaluate_at)

    scenarios = [("free space", NoCollision())]
    if not args.free_only:
        scenarios.append(("blocked (joint-space slab)", JointSlab()))
    if args.metric_sampling:
        print("(metric-aware sampling enabled for the kinetic planner)")
    if args.geodesic_extension:
        print("(geodesic extension enabled for the kinetic planner)")

    rows_by_scenario = {}
    for name, collision in scenarios:
        solver = GafroIKSolver(model.manipulator, model.joint_limits, max_iterations=200,
                               tolerance=1e-5, base_configuration=model.base_configuration,
                               collision_checker=collision)
        print()
        print("=" * 72)
        print(f"scenario: {name}")
        print("=" * 72)
        _model, _kinetic, rows = run_comparison(
            args.queries, args.seed, args.step_size, args.out_dir,
            collision=collision, model=model, kinetic=kinetic, solver=solver,
            metric_sampling=args.metric_sampling,
            geodesic_extension=args.geodesic_extension)
        rows_by_scenario[name] = rows
        report(model, kinetic, rows)

    print()
    if args.geodesic_extension:
        print("Extending along the natural gradient (rather than the straight line)")
        print("is what turns the metric into lower-cost paths.")
    else:
        print("The metric alone mostly changes which node is nearest, and CBiRRT's")
        print("trees connect in ~1 iteration, so there is little growth to steer.")
        print("Re-run with --geodesic-extension to trace each edge along the metric.")

    if args.viz:
        first = next((r for r in rows_by_scenario.values() if r), [])
        visualize(model, first, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
