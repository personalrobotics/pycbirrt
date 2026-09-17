# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for solving a PlanningProblem directly, without TSRs."""

import numpy as np
import pytest

from pycbirrt import (
    AllGoalConfigurationsInvalid,
    AllStartConfigurationsInCollision,
    CBiRRT,
    CBiRRTConfig,
    PlanningProblem,
    UnsupportedCapability,
)
from pycbirrt.sets import AllOf, AnyOf, EmptySet, FiniteSet, PredicateSet, is_finite, members
from tests.test_planner import MockCollisionChecker, MockIKSolver, MockRobotModel


@pytest.fixture
def planner():
    robot = MockRobotModel()
    collision = MockCollisionChecker()
    return CBiRRT(robot, MockIKSolver(robot, collision), collision)


def problem(planner, start, goal, path_constraint=None, validator=None):
    return PlanningProblem(
        space=planner.space,
        start=start,
        goal=goal,
        validator=validator or planner.collision,
        path_constraint=path_constraint,
    )


class TestSolve:
    def test_finite_to_finite(self, planner):
        q0, q1 = np.array([0.0, 0.0]), np.array([1.0, 0.5])
        result = planner.solve(problem(planner, FiniteSet([q0]), FiniteSet([q1])), seed=42)
        assert result.success
        assert np.allclose(result.path[0], q0)
        assert np.allclose(result.path[-1], q1)
        assert result.start_source == (0,)
        assert result.goal_source == (0,)
        assert result.start_index == 0 and result.goal_index == 0

    def test_anyof_goal_reports_which_alternative(self, planner):
        start = FiniteSet([np.array([0.0, 0.0])])
        near, far = np.array([0.1, 0.1]), np.array([3.0, 3.0])
        goal = AnyOf([FiniteSet([far]), FiniteSet([near])])
        result = planner.solve(problem(planner, start, goal), seed=1)
        assert result.success
        assert result.goal_source == (1, 0)
        assert result.goal_index == 0
        assert np.allclose(result.path[-1], near)

    def test_rejection_only_path_constraint(self, planner):
        """A membership-only constraint is enforced by rejection: the path never leaves it."""
        half = PredicateSet(lambda q: q[1] >= -1e-9, name="q1 >= 0")
        q0, q1 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        cfg = CBiRRTConfig(smooth_path=False)
        p = CBiRRT(planner.robot, planner.ik, planner.collision, cfg)
        result = p.solve(problem(p, FiniteSet([q0]), FiniteSet([q1]), path_constraint=half), seed=3)
        assert result.success
        assert all(half.contains(q) for q in result.path)

    def test_root_outside_path_constraint_is_invalid(self, planner):
        half = PredicateSet(lambda q: q[1] > 0.5)
        q0, q1 = np.array([0.0, 1.0]), np.array([1.0, 0.0])  # goal violates
        with pytest.raises(AllGoalConfigurationsInvalid, match="violates path constraints"):
            planner.solve(problem(planner, FiniteSet([q0]), FiniteSet([q1]), path_constraint=half))

    def test_membership_only_start_is_unsupported(self, planner):
        start = PredicateSet(lambda q: True)
        with pytest.raises(UnsupportedCapability, match="start set must be finite or sampleable"):
            planner.solve(problem(planner, start, FiniteSet([np.zeros(2)])))

    def test_empty_goal_raises_legacy_message(self, planner):
        with pytest.raises(ValueError, match="No valid goal configurations available"):
            planner.solve(problem(planner, FiniteSet([np.zeros(2)]), EmptySet()))

    def test_all_roots_in_collision(self, planner):
        class Blocking:
            def is_valid(self, q):
                return False

        with pytest.raises(AllStartConfigurationsInCollision):
            planner.solve(problem(planner, FiniteSet([np.zeros(2)]), FiniteSet([np.ones(2)]), validator=Blocking()))

    def test_finite_intersection_roots_are_filtered(self, planner):
        """AllOf of a finite set and a predicate enumerates only the members the predicate keeps."""
        q_ok, q_bad = np.array([0.0, 0.5]), np.array([0.0, -0.5])
        goal = AllOf([FiniteSet([q_bad, q_ok]), PredicateSet(lambda q: q[1] > 0)])
        assert is_finite(goal)
        assert [tuple(m.source) for m in members(goal)] == [(1,)]
        result = planner.solve(problem(planner, FiniteSet([np.zeros(2)]), goal), seed=0)
        assert result.success
        assert np.allclose(result.path[-1], q_ok)
        assert result.goal_index == 1


class TestEnumeration:
    def test_nested_members_carry_provenance(self):
        a, b, c = (np.array([float(i), 0.0]) for i in range(3))
        s = AnyOf([FiniteSet([a]), AnyOf([FiniteSet([b]), FiniteSet([c])])])
        assert is_finite(s)
        assert [m.source for m in members(s)] == [(0, 0), (1, 0, 0), (1, 1, 0)]

    def test_union_with_infinite_child_is_not_finite(self):
        s = AnyOf([FiniteSet([np.zeros(2)]), PredicateSet(lambda q: True)])
        assert not is_finite(s)
        assert members(s) == []

    def test_empty_set(self):
        assert is_finite(EmptySet())
        assert members(EmptySet()) == []
        assert not EmptySet().contains(np.zeros(2))


class TestRootSampling:
    def test_collision_free_branch_is_found_even_when_not_first(self, planner):
        """A draw yields several candidates; the valid one may not be first (regression).

        The planar arm has two IK branches per pose. Reject the branch the
        solver lists first for every pose; the planner must still seed roots
        from the second branch rather than reporting all samples in collision.
        """
        from tsr import TSR

        from pycbirrt.tsr_set import TSRConfigurationSet

        robot, ik = planner.robot, planner.ik
        first_branch = []

        class RejectFirstBranch:
            def is_valid(self, q):
                return not any(np.allclose(q, f) for f in first_branch)

        class RecordingIK:
            def solve(self, pose, q_init=None):
                sols = ik.solve(pose, q_init)
                if sols:
                    first_branch.append(np.array(sols[0]))
                return sols

        T0_w = np.eye(4)
        T0_w[0, 3], T0_w[1, 3] = 1.2, 0.8
        box = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
        goal = TSRConfigurationSet(TSR(T0_w=T0_w, Tw_e=np.eye(4), Bw=box), robot, RecordingIK(), planner.space)
        validator = RejectFirstBranch()
        result = planner.solve(
            problem(planner, FiniteSet([np.array([0.3, 0.9])]), goal, validator=validator),
            seed=0,
        )
        assert result.success
        assert validator.is_valid(result.path[-1])
        assert goal.contains(result.path[-1])

    def test_max_ik_per_pose_caps_roots_per_draw(self, planner):
        """At most max_ik_per_pose admissible candidates of one draw become roots."""

        class ManyCandidates:
            def contains(self, q):
                return True

            def sample(self, rng):
                base = rng.uniform(-1, 1, 2)
                return [FiniteSet([base + 0.01 * k]).sample(rng)[0] for k in range(5)]

        cfg = CBiRRTConfig(max_ik_per_pose=2, num_tree_roots=4, tsr_samples=10)
        p = CBiRRT(planner.robot, planner.ik, planner.collision, cfg)
        roots = p._roots(problem(p, ManyCandidates(), ManyCandidates()), ManyCandidates(), "Start")
        assert len(roots) == 4
        # Four roots from two draws of two, not one draw of four
        assert not np.allclose(roots[0].q + 0.02, roots[2].q)
