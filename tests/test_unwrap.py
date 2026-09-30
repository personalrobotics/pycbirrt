# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Returned paths are continuous in raw joint values on angular joints (#77).

The planner measures with wraparound on angular joints, so two stored nodes
can be a full turn apart in raw value while a step apart physically. The
returned path is unwrapped forward from its first waypoint.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sscbirrt import CBiRRT, CBiRRTConfig, PlanningProblem
from sscbirrt.sets import FiniteSet
from sscbirrt.space import JointSpace
from tests.test_planner import MockCollisionChecker, MockIKSolver, MockRobotModel

LIMITS = (np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi]))


def make_planner(angular, **cfg):
    robot = MockRobotModel()
    collision = MockCollisionChecker()
    return CBiRRT(robot, MockIKSolver(robot, collision), collision, CBiRRTConfig(continuous_joints=angular, **cfg))


def problem(planner, q0, q1):
    return PlanningProblem(
        space=planner.space, start=FiniteSet([q0]), goal=FiniteSet([q1]), validator=planner.collision
    )


def max_raw_step(path):
    return float(np.abs(np.diff(np.array(path), axis=0)).max())


class TestUnwrapPath:
    def test_no_angular_joints_is_identity(self):
        space = JointSpace(*LIMITS)
        path = [np.array([3.0, 0.0]), np.array([-3.0, 0.1])]
        out = space.unwrap_path(path)
        assert all(np.array_equal(a, b) for a, b in zip(out, path))

    def test_angular_joint_takes_the_short_way(self):
        space = JointSpace(*LIMITS, continuous_joints=(True, False))
        path = [np.array([3.0, 0.0]), np.array([-3.0, 0.1])]
        out = space.unwrap_path(path)
        assert np.array_equal(out[0], path[0])
        assert out[1][0] == pytest.approx(3.0 + 0.283, abs=1e-3)  # +0.28 past pi, not -6 the long way
        assert out[1][1] == pytest.approx(0.1)  # linear joint untouched
        assert space.distance(out[1], path[1]) < 1e-12  # same configuration

    def test_short_paths(self):
        space = JointSpace(*LIMITS, continuous_joints=(True, True))
        assert space.unwrap_path([]) == []
        q = np.array([1.0, 2.0])
        assert np.array_equal(space.unwrap_path([q])[0], q)

    def test_does_not_mutate_input(self):
        space = JointSpace(*LIMITS, continuous_joints=(True, True))
        path = [np.array([3.0, 3.0]), np.array([-3.0, -3.0])]
        copies = [p.copy() for p in path]
        space.unwrap_path(path)
        assert all(np.array_equal(a, b) for a, b in zip(path, copies))

    @settings(max_examples=200, deadline=None)
    @given(
        st.lists(st.tuples(st.floats(-3, 3), st.floats(-3, 3)), min_size=2, max_size=8),
        st.tuples(st.booleans(), st.booleans()),
    )
    def test_laws(self, points, angular):
        """Physical configurations, wrapped distances, and non-angular joints are preserved; angular steps are short."""
        space = JointSpace(*LIMITS, continuous_joints=angular if any(angular) else None)
        path = [np.array(p) for p in points]
        out = space.unwrap_path(path)
        assert len(out) == len(path)
        assert np.array_equal(out[0], path[0])
        for a, b in zip(out, path):
            assert space.distance(a, b) < 1e-9
        for a, b in zip(out[:-1], out[1:]):
            raw = np.abs(b - a)
            for j in range(2):
                if any(angular) and angular[j]:
                    assert raw[j] <= np.pi + 1e-9  # short way around
        for a, b in zip(out, path):
            for j in range(2):
                if not (any(angular) and angular[j]):
                    assert a[j] == b[j]  # non-angular joints untouched


class TestPlannerOutput:
    @pytest.mark.parametrize("smooth", [False, True])
    def test_across_the_seam_is_short_and_continuous(self, smooth):
        planner = make_planner((True, False), step_size=0.2, smooth_path=smooth)
        q0, q1 = np.array([3.0, 0.0]), np.array([-3.0, 0.0])  # 0.28 apart across the seam
        result = planner.solve(problem(planner, q0, q1), seed=0)
        assert result.success
        assert np.array_equal(result.path[0], q0)
        assert planner.space.distance(result.path[-1], q1) < 1e-9
        assert max_raw_step(result.path) <= 0.2 + 1e-9

    def test_goal_given_a_turn_away_comes_back_adjacent(self):
        """Start and goal that differ by exactly 2π on an angular joint are the same configuration."""
        planner = make_planner((True, True), step_size=0.1, smooth_path=False)
        q0 = np.array([0.5, 0.5])
        q1 = q0 + np.array([2 * np.pi, 0.0])
        assert planner.space.contains(q1)
        result = planner.solve(problem(planner, q0, q1), seed=0)
        assert result.success
        assert len(result.path) == 1 or max_raw_step(result.path) <= 0.1 + 1e-9
        assert np.allclose(result.path[-1], q0)

    @pytest.mark.parametrize("seed", range(4))
    def test_every_consecutive_raw_step_within_step_size(self, seed):
        planner = make_planner((True, True), step_size=0.2, connection_tolerance=0.05)
        q0, q1 = np.array([2.9, -2.9]), np.array([-2.9, 2.9])
        result = planner.solve(problem(planner, q0, q1), seed=seed)
        assert result.success
        assert max_raw_step(result.path) <= 0.2 + 1e-9
        assert all(planner.space.contains(q) for q in result.path)

    def test_non_angular_paths_unchanged(self):
        """Without angular joints the unwrap is the identity, so output is exactly what extraction produced."""
        planner = make_planner(None, step_size=0.2, smooth_path=False)
        q0, q1 = np.zeros(2), np.array([1.0, 0.5])
        result = planner.solve(problem(planner, q0, q1), seed=0)
        assert np.array_equal(result.path[0], q0) and np.array_equal(result.path[-1], q1)
        tree_path = result.tree_start.get_path_to_root(len(result.tree_start) - 1)
        assert all(planner.space.contains(q) for q in tree_path)

    def test_legacy_plan_returns_unwrapped_path(self):
        planner = make_planner((True, False), step_size=0.2)
        path = planner.plan(start=np.array([3.0, 0.0]), goal=np.array([-3.0, 0.0]), seed=0)
        assert path is not None and max_raw_step(path) <= 0.2 + 1e-9
