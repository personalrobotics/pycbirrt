# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""End-to-end planning with composed sets: AnyOf goals and AllOf path constraints."""

import numpy as np
import pytest
from tsr import TSR

from pycbirrt import CBiRRT, CBiRRTConfig, PlanningProblem
from pycbirrt.sets import AllOf, AnyOf, FiniteSet, MostViolatedProjection, PredicateSet, RejectionSampling
from pycbirrt.tsr_set import TSRConfigurationSet, tsr_weights
from tests.test_planner import MockCollisionChecker, MockIKSolver, MockRobotModel

BOX = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])


def frame(x, y):
    T = np.eye(4)
    T[0, 3], T[1, 3] = x, y
    return T


@pytest.fixture
def arm():
    robot = MockRobotModel()
    collision = MockCollisionChecker()
    ik = MockIKSolver(robot, collision)
    planner = CBiRRT(robot, ik, collision, CBiRRTConfig(smooth_path=False))
    return robot, ik, collision, planner


def tsr_set(arm, x, y, bw=BOX):
    robot, ik, _, planner = arm
    return TSRConfigurationSet(TSR(T0_w=frame(x, y), Tw_e=np.eye(4), Bw=bw), robot, ik, planner.space)


def problem(arm, start, goal, path_constraint=None):
    _, _, collision, planner = arm
    return PlanningProblem(
        space=planner.space, start=start, goal=goal, validator=collision, path_constraint=path_constraint
    )


class TestAnyOfGoal:
    def test_goal_is_union_of_tsr_alternatives(self, arm):
        """Plan to either of two grasp regions; the result says which one was reached."""
        robot, _, _, planner = arm
        near = tsr_set(arm, 1.9, 0.3)  # reachable near q = 0
        far = tsr_set(arm, -1.5, 1.0)
        goal = AnyOf([far, near], weights=tsr_weights([far, near]))
        result = planner.solve(problem(arm, FiniteSet([np.zeros(2)]), goal), seed=0)
        assert result.success
        (which,) = result.goal_source
        assert [far, near][which].contains(result.path[-1])
        assert goal.contains(result.path[-1])
        assert result.goal_index == which

    def test_nested_matched_alternatives(self, arm):
        """(A and P1) or (B and P2): grouping is preserved in what the planner accepts."""
        A = tsr_set(arm, 1.9, 0.3)
        B = tsr_set(arm, 0.3, 1.9)
        upper = PredicateSet(lambda q: q[0] > 0.5, name="q0 > 0.5")  # B's configs satisfy this, A's do not
        lower = PredicateSet(lambda q: q[0] <= 0.5, name="q0 <= 0.5")
        by_rejection = RejectionSampling(source=0)
        matched = AnyOf(
            [AllOf([A, lower], sampling=by_rejection), AllOf([B, upper], sampling=by_rejection)], weights=[1, 1]
        )
        crossed = AnyOf(
            [AllOf([A, upper], sampling=by_rejection), AllOf([B, lower], sampling=by_rejection)], weights=[1, 1]
        )

        # The matched pairing has members; the crossed pairing is (nearly) empty for this arm
        rng = np.random.default_rng(0)
        assert any(matched.contains(c.q) for _ in range(50) for c in matched.sample(rng))
        assert not any(crossed.contains(c.q) for _ in range(50) for c in crossed.sample(rng))

        result = arm[3].solve(problem(arm, FiniteSet([np.zeros(2)]), matched), seed=1)
        assert result.success
        assert matched.contains(result.path[-1])
        assert not crossed.contains(result.path[-1])

    def test_mixed_finite_and_tsr_alternatives(self, arm):
        """A union of a fixed configuration and a TSR region, with a finite start."""
        fixed = FiniteSet([np.array([0.4, 0.2])])
        region = tsr_set(arm, -1.5, 1.0)
        goal = AnyOf([fixed, region], weights=[0.0, 1.0])
        result = arm[3].solve(problem(arm, FiniteSet([np.zeros(2)]), goal), seed=2)
        assert result.success
        assert goal.contains(result.path[-1])
        assert result.goal_source[0] in (0, 1)


class TestAllOfPathConstraint:
    def test_intersection_of_two_tsrs_along_the_path(self, arm):
        """Every waypoint satisfies both constraint TSRs, projected with the named strategy."""
        robot, _, _, planner = arm
        # Two overlapping bands: EE x in [1.0, 2.0] and EE y in [-0.6, 0.6] (loose in the other axis)
        band_x = np.array([[-0.5, 0.5], [-2.0, 2.0], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
        band_y = np.array([[-2.0, 2.0], [-0.6, 0.6], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
        x_band = tsr_set(arm, 1.5, 0.0, band_x)
        y_band = tsr_set(arm, 0.0, 0.0, band_y)
        constraint = AllOf([x_band, y_band], projection=MostViolatedProjection())

        q_start = np.array([0.0, 0.6])
        q_goal = np.array([-0.3, 0.9])
        for q in (q_start, q_goal):
            assert constraint.contains(q), robot.forward_kinematics(q)[:2, 3]

        result = planner.solve(problem(arm, FiniteSet([q_start]), FiniteSet([q_goal]), constraint), seed=3)
        assert result.success
        for q in result.path:
            assert x_band.contains(q) and y_band.contains(q)

    def test_intersection_with_rejection_only_child_is_enforced(self, arm):
        """AllOf(TSR, predicate) has no projector; membership is enforced by rejection."""
        _, _, _, planner = arm
        band = np.array([[-2.0, 2.0], [-0.6, 0.6], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
        y_band = tsr_set(arm, 0.0, 0.0, band)
        elbow_up = PredicateSet(lambda q: q[1] >= 0.0, name="elbow up")
        constraint = AllOf([y_band, elbow_up])
        q_start, q_goal = np.array([0.0, 0.6]), np.array([-0.3, 0.9])
        result = planner.solve(problem(arm, FiniteSet([q_start]), FiniteSet([q_goal]), constraint), seed=4)
        assert result.success
        for q in result.path:
            assert constraint.contains(q)

    def test_unsatisfiable_intersection_reports_invalid_roots(self, arm):
        from pycbirrt import AllStartConfigurationsInvalid

        never = PredicateSet(lambda q: False)
        constraint = AllOf([tsr_set(arm, 1.5, 0.0), never])
        with pytest.raises(AllStartConfigurationsInvalid):
            arm[3].solve(problem(arm, FiniteSet([np.array([0.0, 0.6])]), FiniteSet([np.array([0.0, 0.6])]), constraint))
