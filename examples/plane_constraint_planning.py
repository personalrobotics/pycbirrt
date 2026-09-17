# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Example: CBiRRT planning with a PlaneConstraint path constraint.

Demonstrates the new geometric-primitive constraint (``tsr.PlaneConstraint``)
as a drop-in path constraint for the planner. The 2-DOF planar arm lives in the
z=0 plane, so a world plane whose normal lies in the xy-plane intersects it in a
*line* y = tan(theta) * x + b. Constraining the end-effector to that plane forces
the whole trajectory onto an arbitrary, possibly tilted, line in the workspace --
the geometric-primitive analogue of the axis-aligned y-band in ``planar_arm.py``.

Requires only numpy, matplotlib, and tsr (no MuJoCo / EAIK). Reuses the planar
arm robot/IK/collision helpers from ``planar_arm.py``.

Run with:
    python examples/plane_constraint_planning.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

# Reuse the planar-arm helpers and the position-TSR factory from the sibling example.
from planar_arm import (  # noqa: E402  (local example import)
    CircleObstacleChecker,
    PlanarArmIK,
    PlanarArmRobot,
    make_position_tsr,
    visualize_result,
)
from tsr import PlaneConstraint

from pycbirrt import CBiRRT, CBiRRTConfig


def make_line_plane_constraint(theta: float, offset: float) -> PlaneConstraint:
    """Constrain the end-effector to the workspace line y = tan(theta)*x + offset.

    The planar arm's FK output is a pose at (x, y, 0). A world plane that contains
    the z axis direction and whose in-xy normal is perpendicular to the line
    direction ``(cos theta, sin theta, 0)`` intersects the z=0 plane exactly in
    that line. The normal is ``(-sin theta, cos theta, 0)`` and the plane passes
    through the point ``(0, offset, 0)``.
    """
    normal = [-np.sin(theta), np.cos(theta), 0.0]
    point = [0.0, offset, 0.0]
    return PlaneConstraint.from_point_normal(point, normal)


def line_endpoints(theta: float, offset: float, span: float = 2.0):
    """Two workspace points on the constraint line, for placing start/goal regions."""
    d = np.array([np.cos(theta), np.sin(theta)])
    mid = np.array([0.0, offset])
    return mid - 0.5 * span * d, mid + 0.5 * span * d


def example_tilted_line(theta_deg: float = 30.0, offset: float = 0.3):
    """Plan along a tilted line constraint."""
    print("\n" + "=" * 60)
    print(f"Plane constraint planning: line at {theta_deg} deg, offset {offset}")
    print("=" * 60)

    theta = np.radians(theta_deg)
    robot = PlanarArmRobot(l1=1.0, l2=1.0)
    obstacles = []
    collision = CircleObstacleChecker(robot, obstacles)
    ik = PlanarArmIK(robot, collision)

    config = CBiRRTConfig(
        step_size=0.2,
        goal_bias=0.1,
        smooth_path=True,
        smoothing_iterations=50,
        # Allowed distance from the constraint plane; keep tight for a crisp line.
        tsr_tolerance=0.01,
        angular_joints=(True, True),
    )
    planner = CBiRRT(robot, ik, collision, config)

    constraint = make_line_plane_constraint(theta, offset)
    print(f"  Constraint: {constraint}")

    # Start/goal regions placed on the line so the endpoints already satisfy it.
    p_start, p_goal = line_endpoints(theta, offset, span=2.0)
    start_tsr = make_position_tsr(p_start[0], p_start[1], tolerance=0.05)
    goal_tsr = make_position_tsr(p_goal[0], p_goal[1], tolerance=0.05)
    print(f"  Start region: ({p_start[0]:.2f}, {p_start[1]:.2f})")
    print(f"  Goal region:  ({p_goal[0]:.2f}, {p_goal[1]:.2f})")

    result = planner.plan(
        start=None,
        goal_tsrs=[goal_tsr],
        start_tsrs=[start_tsr],
        constraint_tsrs=[constraint],
        seed=42,
        return_details=True,
    )

    print(f"  Iterations: {result.iterations}")
    print(f"  Tree sizes: start={result.tree_sizes[0]}, goal={result.tree_sizes[1]}")

    if not result.success:
        print("  No path found!")
        print(f"  Reason: {result.failure_reason}")
        return

    print(f"  Found path with {len(result.path)} waypoints")

    # Verify every waypoint lies on the constraint plane.
    max_violation = max(constraint.distance(robot.forward_kinematics(q))[0] for q in result.path)
    print(f"  Max distance off the constraint plane: {max_violation:.4f}")

    _visualize_line(robot, result.path, obstacles, collision, theta, offset, (p_start, 0.05), (p_goal, 0.05))


def _visualize_line(robot, path, obstacles, collision, theta, offset, start_region, goal_region):
    """Workspace + C-space plot, with the constraint line overlaid (see planar_arm.visualize_result)."""
    # Draw the standard two-panel result, then overlay the constraint line on the
    # workspace axis.
    visualize_result(
        robot,
        path,
        obstacles,
        collision,
        start_region=start_region,
        goal_region=goal_region,
        title=f"PlaneConstraint Planning (line at {np.degrees(theta):.0f} deg)",
        filename="plane_constraint_result.png",
    )
    # visualize_result already saved/showed; add the line to a fresh quick plot so
    # the constraint is unmistakable.
    d = np.array([np.cos(theta), np.sin(theta)])
    mid = np.array([0.0, offset])
    a, b = mid - 1.5 * d, mid + 1.5 * d
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([a[0], b[0]], [a[1], b[1]], "y-", linewidth=6, alpha=0.4, label="Constraint line (plane ∩ z=0)")
    xs = [robot.forward_kinematics(q)[0, 3] for q in path]
    ys = [robot.forward_kinematics(q)[1, 3] for q in path]
    ax.plot(xs, ys, "k.-", label="End-effector path")
    ax.scatter([xs[0]], [ys[0]], c="blue", s=150, marker="*", zorder=5, label="Start")
    ax.scatter([xs[-1]], [ys[-1]], c="green", s=150, marker="*", zorder=5, label="Goal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title("End-effector stays on the constraint line")
    plt.tight_layout()
    plt.savefig("plane_constraint_line.png", dpi=150)
    print("  Saved end-effector trace to plane_constraint_line.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="CBiRRT plane-constraint planning example")
    parser.add_argument("--angle", type=float, default=30.0, help="Line tilt in degrees (default: 30)")
    parser.add_argument("--offset", type=float, default=0.3, help="Line y-offset at x=0 (default: 0.3)")
    args = parser.parse_args()
    example_tilted_line(theta_deg=args.angle, offset=args.offset)


if __name__ == "__main__":
    main()
