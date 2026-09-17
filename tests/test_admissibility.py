# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Every configuration the planner accepts belongs to the problem's joint space (#43).

Adversarial inputs: fixed roots outside limits, malformed arrays, samplers and
projectors that return out-of-space configurations. The invariant under test is
that nothing outside ``problem.space`` is ever stored in a tree or returned.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pycbirrt import (
    AllGoalConfigurationsInvalid,
    AllStartConfigurationsInCollision,
    AllStartConfigurationsInvalid,
    CBiRRT,
    CBiRRTConfig,
    PlanningProblem,
)
from pycbirrt.sets import FiniteSet, Sample
from pycbirrt.space import JointSpace
from pycbirrt.tree import RRTree
from tests.test_planner import MockCollisionChecker, MockIKSolver, MockRobotModel


@pytest.fixture
def planner():
    robot = MockRobotModel()
    collision = MockCollisionChecker()
    return CBiRRT(robot, MockIKSolver(robot, collision), collision, CBiRRTConfig(smooth_path=False))


def problem(planner, start, goal, path_constraint=None, validator=None, space=None):
    return PlanningProblem(
        space=space or planner.space,
        start=start,
        goal=goal,
        validator=validator or planner.collision,
        path_constraint=path_constraint,
    )


def all_nodes_in_space(result, space):
    return all(space.contains(n.config) for tree in (result.tree_start, result.tree_goal) for n in tree.nodes)


# ---------------------------------------------------------------------------
# JointSpace membership
# ---------------------------------------------------------------------------


class TestJointSpaceContains:
    space = JointSpace(np.array([-1.0, -2.0]), np.array([1.0, 2.0]))

    @pytest.mark.parametrize(
        "q, fragment",
        [
            (np.array([0.0, 0.0, 0.0]), "shape (3,) != (2,)"),
            (np.array([0.0]), "shape (1,) != (2,)"),
            (np.zeros((2, 1)), "shape (2, 1) != (2,)"),
            (np.array([np.nan, 0.0]), "non-finite"),
            (np.array([0.0, np.inf]), "non-finite"),
            (np.array([1.5, 0.0]), "joint 0: 1.5 not in [-1, 1]"),
            (np.array([0.0, -2.5]), "joint 1: -2.5 not in [-2, 2]"),
            ("not a config", "not numeric"),
        ],
    )
    def test_rejects_with_reason(self, q, fragment):
        why = self.space.why_invalid(q)
        assert why is not None and fragment in why
        assert not self.space.contains(q)

    def test_accepts_members_and_boundaries(self):
        for q in (np.zeros(2), np.array([1.0, 2.0]), np.array([-1.0, -2.0]), [0.5, 0.5]):
            assert self.space.why_invalid(q) is None
            assert self.space.contains(q)

    def test_angular_joint_accepts_any_finite_value(self):
        s = JointSpace(np.array([-np.pi, -1.0]), np.array([np.pi, 1.0]), angular_joints=(True, False))
        assert s.contains(np.array([100.0, 0.0]))
        assert not s.contains(np.array([np.inf, 0.0]))
        assert "joint 1" in s.why_invalid(np.array([100.0, 5.0]))
        assert "joint 0" not in s.why_invalid(np.array([100.0, 5.0]))

    @settings(max_examples=300, deadline=None)
    @given(
        st.lists(
            st.one_of(st.floats(-3, 3), st.just(np.nan), st.just(np.inf), st.just(-np.inf)),
            min_size=0,
            max_size=4,
        )
    )
    def test_contains_agrees_with_definition(self, values):
        q = np.array(values, dtype=float)
        expected = (
            q.shape == (2,)
            and bool(np.all(np.isfinite(q)))
            and bool(np.all((q >= self.space.lower) & (q <= self.space.upper)))
        )
        assert self.space.contains(q) == expected


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


class TestRootsOutsideSpace:
    def test_fixed_start_outside_limits_is_invalid_not_collision(self, planner):
        with pytest.raises(AllStartConfigurationsInvalid, match="outside joint space") as exc:
            planner.solve(problem(planner, FiniteSet([np.array([10.0, 0.0])]), FiniteSet([np.zeros(2)])))
        assert "joint 0" in str(exc.value)
        assert not isinstance(exc.value, AllStartConfigurationsInCollision)

    def test_fixed_goal_outside_limits_is_invalid(self, planner):
        with pytest.raises(AllGoalConfigurationsInvalid, match="outside joint space"):
            planner.solve(problem(planner, FiniteSet([np.zeros(2)]), FiniteSet([np.array([0.0, -10.0])])))

    @pytest.mark.parametrize(
        "bad",
        [np.array([0.0, 0.0, 0.0]), np.array([np.nan, 0.0]), np.array([np.inf, 0.0])],
        ids=["shape", "nan", "inf"],
    )
    def test_malformed_fixed_start_is_rejected_clearly(self, planner, bad):
        with pytest.raises(AllStartConfigurationsInvalid, match="outside joint space"):
            planner.solve(problem(planner, FiniteSet([bad]), FiniteSet([np.zeros(2)])))

    def test_legacy_plan_rejects_out_of_limit_start(self, planner):
        with pytest.raises(AllStartConfigurationsInvalid, match="outside joint space"):
            planner.plan(start=np.array([10.0, 0.0]), goal=np.zeros(2))

    def test_valid_and_invalid_fixed_roots_mixed(self, planner, caplog):
        import logging

        starts = [np.array([10.0, 0.0]), np.array([0.2, 0.2])]
        with caplog.at_level(logging.WARNING):
            result = planner.solve(problem(planner, FiniteSet(starts), FiniteSet([np.zeros(2)])), seed=0)
        assert result.success
        assert "Start[0]: outside joint space" in caplog.text
        assert all_nodes_in_space(result, planner.space)

    def test_sampler_returning_out_of_space_candidates(self, planner):
        class BadSampler:
            def contains(self, q):
                return True

            def sample(self, rng):
                return [Sample(np.array([10.0, 0.0])), Sample(np.array([np.nan, 0.0])), Sample(np.zeros(3))]

        with pytest.raises(AllGoalConfigurationsInvalid, match="outside joint space") as exc:
            planner.solve(problem(planner, FiniteSet([np.zeros(2)]), BadSampler()))
        assert "in collision" not in str(exc.value)

    def test_sampler_mixing_good_and_bad_candidates(self, planner):
        class MixedSampler:
            def contains(self, q):
                return True

            def sample(self, rng):
                return [Sample(np.array([10.0, 0.0])), Sample(np.array([0.3, 0.1]))]

        cfg = CBiRRTConfig(smooth_path=False, num_tree_roots=5, tsr_samples=20)
        p = CBiRRT(planner.robot, planner.ik, planner.collision, cfg)
        result = p.solve(problem(p, FiniteSet([np.zeros(2)]), MixedSampler()), seed=0)
        assert result.success
        assert all_nodes_in_space(result, p.space)
        assert np.allclose(result.path[-1], [0.3, 0.1])

    def test_only_collisions_still_reports_collision(self, planner):
        """Sampled goal candidates all in collision (and in space) keep the collision exception."""
        from pycbirrt import AllGoalConfigurationsInCollision

        class BlockRight:
            def is_valid(self, q):
                return q[0] < 0.5

        class RightSampler:
            def contains(self, q):
                return True

            def sample(self, rng):
                return [Sample(np.array([rng.uniform(0.6, 0.9), 0.0]))]

        with pytest.raises(AllGoalConfigurationsInCollision, match="in collision"):
            planner.solve(problem(planner, FiniteSet([np.zeros(2)]), RightSampler(), validator=BlockRight()))

    def test_mixed_collision_and_out_of_space_is_invalid(self, planner):
        """If any rejection was for being outside the space, the exception is Invalid, not InCollision."""
        from pycbirrt import AllGoalConfigurationsInCollision

        class BlockRight:
            def is_valid(self, q):
                return q[0] < 0.5

        class Alternating:
            def contains(self, q):
                return True

            def sample(self, rng):
                return [Sample(np.array([0.7, 0.0])), Sample(np.array([10.0, 0.0]))]

        with pytest.raises(AllGoalConfigurationsInvalid, match="outside joint space") as exc:
            planner.solve(problem(planner, FiniteSet([np.zeros(2)]), Alternating(), validator=BlockRight()))
        assert "in collision" in str(exc.value)
        assert not isinstance(exc.value, AllGoalConfigurationsInCollision)


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------


def quick(planner):
    """Same planner with a small budget, for problems the adversarial sets make unsolvable."""
    cfg = CBiRRTConfig(smooth_path=False, timeout=1.0, max_iterations=500)
    return CBiRRT(planner.robot, planner.ik, planner.collision, cfg)


class TestExtensionsOutsideSpace:
    def test_projector_moving_outside_space_is_rejected(self, planner):
        """A projector that returns out-of-space points must not get them into the tree."""
        planner = quick(planner)

        class Escaping:
            calls = 0

            def contains(self, q):
                return True

            def project(self, q_prev, q):
                Escaping.calls += 1
                return np.array([10.0, 0.0]) if q[0] > 0.2 else q

        result = planner.solve(
            problem(planner, FiniteSet([np.zeros(2)]), FiniteSet([np.array([0.9, 0.0])]), path_constraint=Escaping()),
            seed=0,
        )
        assert Escaping.calls > 0
        assert all_nodes_in_space(result, planner.space)
        if result.success:
            assert all(planner.space.contains(q) for q in result.path)

    def test_projector_returning_malformed_point_is_rejected(self, planner):
        class Malformed:
            def contains(self, q):
                return True

            def project(self, q_prev, q):
                return np.array([np.nan, np.nan])

        planner = quick(planner)
        result = planner.solve(
            problem(planner, FiniteSet([np.zeros(2)]), FiniteSet([np.array([0.9, 0.0])]), path_constraint=Malformed()),
            seed=0,
        )
        assert not result.success
        assert all_nodes_in_space(result, planner.space)

    def test_grow_rejects_projected_endpoint_outside_space(self, planner):
        class Escaping:
            def contains(self, q):
                return True

            def project(self, q_prev, q):
                return np.array([10.0, 0.0])

        prob = problem(planner, FiniteSet([np.zeros(2)]), FiniteSet([np.ones(2)]), path_constraint=Escaping())
        tree = RRTree(np.zeros(2))
        idx, reached = planner._grow(prob, tree, np.array([0.5, 0.0]))
        assert not reached and len(tree) == 1

    def test_every_tree_node_in_space_after_random_plans(self, planner):
        """End-to-end invariant over several seeds on an unconstrained problem."""
        for seed in range(5):
            result = planner.solve(
                problem(planner, FiniteSet([np.zeros(2)]), FiniteSet([np.array([1.0, 0.5])])), seed=seed
            )
            assert result.success
            assert all_nodes_in_space(result, planner.space)

    def test_admissible_reports_space_before_collision(self, planner):
        class Blocking:
            def is_valid(self, q):
                return False

        prob = problem(planner, FiniteSet([np.zeros(2)]), FiniteSet([np.zeros(2)]), validator=Blocking())
        ok, reason = planner._admissible(prob, np.array([10.0, 0.0]))
        assert not ok and reason.startswith("outside joint space")
        ok, reason = planner._admissible(prob, np.zeros(2))
        assert not ok and reason == "in collision"
