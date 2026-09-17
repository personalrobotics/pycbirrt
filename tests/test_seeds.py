# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Explicit configurations embedded in a set are always tree roots (#42).

``seeds`` is distinct from ``members``: a union of a finite set and a
sampleable region is not finite, but its finite part still seeds the search.
Mixture weights govern sampling only.
"""

import logging

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tsr import TSR

from pycbirrt import AllStartConfigurationsInvalid, CBiRRT, CBiRRTConfig, PlanningProblem
from pycbirrt.legacy import legacy_problem
from pycbirrt.sets import AllOf, AnyOf, EmptySet, FiniteSet, PredicateSet, Sample, is_finite, members, seeds
from tests.test_planner import MockCollisionChecker, MockIKSolver, MockRobotModel

BOX = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])


def frame(x, y):
    T = np.eye(4)
    T[0, 3], T[1, 3] = x, y
    return T


class Deterministic:
    """A sampleable, non-finite set that always returns the same configuration."""

    def __init__(self, q):
        self.q = np.asarray(q, dtype=float)

    def contains(self, q):
        return True

    def sample(self, rng):
        return [Sample(self.q.copy())]


@pytest.fixture
def planner():
    robot = MockRobotModel()
    collision = MockCollisionChecker()
    return CBiRRT(robot, MockIKSolver(robot, collision), collision, CBiRRTConfig(smooth_path=False, num_tree_roots=5))


def problem(planner, start, goal, validator=None):
    return PlanningProblem(space=planner.space, start=start, goal=goal, validator=validator or planner.collision)


def lowered(planner, **kw):
    return legacy_problem(
        planner.robot,
        planner.ik,
        planner.collision,
        planner.space,
        planner.config,
        kw.get("start"),
        kw.get("goal"),
        kw.get("start_tsrs"),
        kw.get("goal_tsrs"),
        kw.get("constraint_tsrs"),
    )


def has_root(roots, q):
    return any(np.allclose(r.q, q) for r in roots)


# ---------------------------------------------------------------------------
# seeds() semantics
# ---------------------------------------------------------------------------


class TestSeeds:
    def test_issue_repro_mixed_union_keeps_fixed_root(self, planner):
        fixed = FiniteSet([np.array([0.0, 0.0])])
        sampled = Deterministic([1.0, 1.0])
        role = AnyOf([fixed, sampled], weights=[0.0, 1.0])
        assert not is_finite(role)
        assert members(role) == []
        assert [s.source for s in seeds(role)] == [(0, 0)]
        roots = planner._roots(problem(planner, role, role), role, "Start")
        assert has_root(roots, [0.0, 0.0])
        assert has_root(roots, [1.0, 1.0])

    def test_seeds_equal_members_when_finite(self):
        a, b, c = (np.array([float(i), 0.0]) for i in range(3))
        s = AnyOf([FiniteSet([a]), AnyOf([FiniteSet([b]), FiniteSet([c])])])
        assert [m.source for m in members(s)] == [m.source for m in seeds(s)] == [(0, 0), (1, 0, 0), (1, 1, 0)]

    def test_union_seeds_carry_child_index_regardless_of_weight(self):
        s = AnyOf([Deterministic([9.0, 9.0]), FiniteSet([np.zeros(2)])], weights=[1.0, 0.0])
        assert [x.source for x in seeds(s)] == [(1, 0)]

    def test_intersection_filters_seeds_by_other_children(self):
        keep, drop = np.array([0.0, 0.5]), np.array([0.0, -0.5])
        s = AllOf([FiniteSet([drop, keep]), PredicateSet(lambda q: q[1] > 0), Deterministic([0, 0])])
        assert [x.source for x in seeds(s)] == [(1,)]

    def test_intersection_with_no_seed_bearing_child(self):
        assert seeds(AllOf([Deterministic([0, 0]), PredicateSet(lambda q: True)])) == []

    def test_leaves_without_seeds(self):
        assert seeds(Deterministic([0, 0])) == []
        assert seeds(PredicateSet(lambda q: True)) == []
        assert seeds(EmptySet()) == []

    @settings(max_examples=200, deadline=None)
    @given(st.lists(st.floats(-1, 1), min_size=1, max_size=3), st.booleans(), st.booleans())
    def test_seeds_are_members_and_finite_agrees(self, xs, mixed, nested):
        finite = FiniteSet([np.array([x, 0.0]) for x in xs])
        s = finite
        if nested:
            s = AnyOf([s, FiniteSet([np.array([5.0, 5.0])])])
        if mixed:
            s = AnyOf([s, Deterministic([2.0, 2.0])], weights=[1, 1])
        for x in seeds(s):
            assert s.contains(x.q)
        assert is_finite(s) == (not mixed)
        if is_finite(s):
            assert [m.source for m in members(s)] == [m.source for m in seeds(s)]
        else:
            assert members(s) == [] and len(seeds(s)) == len(xs) + (1 if nested else 0)


# ---------------------------------------------------------------------------
# Legacy lowering
# ---------------------------------------------------------------------------


class TestLegacyMixedRoles:
    def tsr(self):
        return TSR(T0_w=frame(1.2, 0.8), Tw_e=np.eye(4), Bw=BOX)

    def test_start_with_start_tsrs_keeps_fixed_start(self, planner):
        q_fixed = np.array([0.3, 0.3])
        prob = lowered(planner, start=[q_fixed], start_tsrs=[self.tsr()], goal=[np.zeros(2)])
        roots = planner._roots(prob, prob.start, "Start")
        assert has_root(roots, q_fixed)
        assert len(roots) > 1  # sampled TSR roots too
        fixed_roots = [r for r in roots if np.allclose(r.q, q_fixed)]
        assert fixed_roots[0].source == (0, 0)
        sampled = [r for r in roots if not np.allclose(r.q, q_fixed)]
        assert all(r.source[0] == 1 for r in sampled)

    def test_goal_with_goal_tsrs_keeps_fixed_goal(self, planner):
        q_fixed = np.array([0.3, 0.3])
        prob = lowered(planner, start=[np.zeros(2)], goal=[q_fixed], goal_tsrs=[self.tsr()])
        roots = planner._roots(prob, prob.goal, "Goal")
        assert has_root(roots, q_fixed)

    def test_plan_seeds_fixed_start_alongside_tsr_roots(self, planner):
        q_fixed = np.array([0.1, 0.0])
        result = planner.plan(start=[q_fixed], start_tsrs=[self.tsr()], goal=np.zeros(2), seed=0, return_details=True)
        assert result.success
        roots = [n for n in result.tree_start.nodes if n.parent is None]
        assert any(np.allclose(n.config, q_fixed) and n.source_index == (0, 0) for n in roots)
        assert any(n.source_index[0] == 1 for n in roots)

    def test_plan_reports_fixed_start_index_when_tsr_unreachable(self, planner):
        """Only the fixed start can be a root; the path starts there and the legacy index is its position."""
        q_fixed = np.array([0.1, 0.0])
        far = TSR(T0_w=frame(10.0, 0.0), Tw_e=np.eye(4), Bw=BOX)
        result = planner.plan(start=[q_fixed], start_tsrs=[far], goal=np.zeros(2), seed=0, return_details=True)
        assert result.success
        assert np.allclose(result.path[0], q_fixed)
        assert result.start_index == 0 and result.start_source == (0, 0)

    def test_two_fixed_starts_and_a_tsr(self, planner):
        a, b = np.array([0.1, 0.0]), np.array([0.2, 0.0])
        prob = lowered(planner, start=[a, b], start_tsrs=[self.tsr()], goal=[np.zeros(2)])
        roots = planner._roots(prob, prob.start, "Start")
        assert has_root(roots, a) and has_root(roots, b)
        assert {r.source for r in roots if r.source[0] == 0} == {(0, 0), (0, 1)}


# ---------------------------------------------------------------------------
# Validation and sampling interplay
# ---------------------------------------------------------------------------


class TestSeedValidation:
    def test_invalid_seed_is_filtered_but_sampled_roots_proceed(self, planner, caplog):
        role = AnyOf([FiniteSet([np.array([10.0, 0.0])]), Deterministic([0.5, 0.5])], weights=[0.0, 1.0])
        with caplog.at_level(logging.WARNING):
            roots = planner._roots(problem(planner, role, role), role, "Start")
        assert has_root(roots, [0.5, 0.5])
        assert not has_root(roots, [10.0, 0.0])
        assert "Start[0]: outside joint space" in caplog.text

    def test_all_seeds_invalid_and_no_samples_reports_both(self, planner):
        class Nothing:
            def contains(self, q):
                return True

            def sample(self, rng):
                return []

        role = AnyOf([FiniteSet([np.array([10.0, 0.0])]), Nothing()], weights=[0.0, 1.0])
        with pytest.raises(AllStartConfigurationsInvalid) as exc:
            planner._roots(problem(planner, role, role), role, "Start")
        assert "outside joint space" in str(exc.value)
        assert "sampling:" in str(exc.value)

    def test_weights_govern_sampling_only(self, planner):
        """A finite child with nonzero weight is seeded once, not duplicated by sampling."""
        fixed = FiniteSet([np.array([0.0, 0.0])])
        role = AnyOf([fixed, Deterministic([1.0, 1.0])], weights=[1.0, 1.0])
        roots = planner._roots(problem(planner, role, role), role, "Start")
        assert sum(np.allclose(r.q, [0.0, 0.0]) for r in roots) == 1
        assert has_root(roots, [1.0, 1.0])

    def test_finite_only_roles_never_sample(self, planner):
        class Spy(FiniteSet):
            draws = 0

            def sample(self, rng):
                Spy.draws += 1
                return super().sample(rng)

        role = Spy([np.zeros(2), np.array([0.1, 0.1])])
        roots = planner._roots(problem(planner, role, role), role, "Start")
        assert len(roots) == 2 and Spy.draws == 0
