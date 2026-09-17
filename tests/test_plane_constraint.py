# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Integration test: PlaneConstraint as a path constraint for CBiRRT.

Uses the 2-DOF planar arm (numpy-only, no MuJoCo/EAIK). The arm's FK lives in
the z=0 plane, so a PlaneConstraint on the world plane ``y = y0`` confines the
end-effector to a horizontal line — the geometric-primitive analogue of the
``make_y_constraint_tsr`` band in examples/planar_arm.py.
"""

import sys
from pathlib import Path

import pytest
from tsr import PlaneConstraint

from pycbirrt import CBiRRT, CBiRRTConfig

# Reuse the example's planar-arm robot/IK/collision helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from planar_arm import CircleObstacleChecker, PlanarArmIK, PlanarArmRobot  # noqa: E402


def _make_planner(tsr_tolerance=0.01):
    robot = PlanarArmRobot(l1=1.0, l2=1.0)
    collision = CircleObstacleChecker(robot, [])
    ik = PlanarArmIK(robot, collision)
    config = CBiRRTConfig(
        step_size=0.2,
        goal_bias=0.1,
        smooth_path=True,
        smoothing_iterations=30,
        tsr_tolerance=tsr_tolerance,
        angular_joints=(True, True),
    )
    return CBiRRT(robot, ik, collision, config), robot


def _y_plane(y0: float) -> PlaneConstraint:
    """World plane ``y = y0`` (normal along +y)."""
    return PlaneConstraint.from_point_normal([0.0, y0, 0.0], [0.0, 1.0, 0.0])


def test_plane_constraint_is_accepted_and_satisfied():
    planner, robot = _make_planner()
    y0 = 1.0
    constraint = _y_plane(y0)

    # Endpoints on the constraint plane (y = y0).
    from planar_arm import make_position_tsr

    start_tsr = make_position_tsr(-1.2, y0, tolerance=0.05)
    goal_tsr = make_position_tsr(1.2, y0, tolerance=0.05)

    result = planner.plan(
        start=None,
        goal_tsrs=[goal_tsr],
        start_tsrs=[start_tsr],
        constraint_tsrs=[constraint],
        seed=7,
        return_details=True,
    )

    assert result.success, f"planning failed: {result.failure_reason}"

    # Every waypoint's end-effector must lie on the plane within tolerance.
    tol = planner.config.tsr_tolerance
    for q in result.path:
        y = robot.forward_kinematics(q)[1, 3]
        assert abs(y - y0) <= tol + 1e-6, f"waypoint off plane: y={y}"


def test_plane_constraint_matches_tsr_band():
    """A thin y-band TSR and a y-plane constraint should both keep y≈y0."""
    planner, robot = _make_planner()
    y0 = 0.9
    from planar_arm import make_position_tsr

    result = planner.plan(
        start=None,
        goal_tsrs=[make_position_tsr(1.0, y0, tolerance=0.05)],
        start_tsrs=[make_position_tsr(-1.0, y0, tolerance=0.05)],
        constraint_tsrs=[_y_plane(y0)],
        seed=3,
        return_details=True,
    )
    assert result.success, f"planning failed: {result.failure_reason}"
    ys = [robot.forward_kinematics(q)[1, 3] for q in result.path]
    assert max(abs(y - y0) for y in ys) <= planner.config.tsr_tolerance + 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
