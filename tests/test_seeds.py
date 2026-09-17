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


# ---------------------------------------------------------------------------
# Finite intersections and multi-child seeds (#55)
# ---------------------------------------------------------------------------


class Region:
    """A sampleable, non-finite half-plane q[0] >= lo, with exact membership (unlike Deterministic)."""

    def __init__(self, lo):
        self.lo = lo

    def contains(self, q):
        return float(q[0]) >= self.lo

    def sample(self, rng):
        return [Sample(np.array([self.lo + 0.5, 0.0]))]


class TestFiniteIntersections:
    def test_issue_repro_enumerates_the_finite_child(self, planner):
        mixed = AnyOf([FiniteSet([np.array([0.0, 0.0])]), PredicateSet(lambda q: q[0] == 1.0)])
        inter = AllOf([mixed, FiniteSet([np.array([1.0, 0.0])])])
        assert is_finite(inter) and inter.contains(np.array([1.0, 0.0]))
        assert [m.q.tolist() for m in members(inter)] == [[1.0, 0.0]]
        assert [m.q.tolist() for m in seeds(inter)] == [[1.0, 0.0]]
        roots = planner._roots(problem(planner, inter, inter), inter, "Start")
        assert has_root(roots, [1.0, 0.0])

    def test_members_configurations_independent_of_child_order(self):
        a = FiniteSet([np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])])
        b = FiniteSet([np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0])])
        p = PredicateSet(lambda q: q[0] < 2.5)
        for order in ([a, b, p], [p, b, a], [b, p, a]):
            got = sorted(m.q.tolist() for m in members(AllOf(order)))
            assert got == [[1.0, 0.0], [2.0, 0.0]]

    def test_non_finite_intersection_collects_seeds_from_every_child(self):
        keep = PredicateSet(lambda q: q[0] >= 0)
        left = AnyOf([FiniteSet([np.array([-1.0, 0.0]), np.array([1.0, 0.0])]), Region(4.0)], weights=[1, 1])
        right = AnyOf([FiniteSet([np.array([2.0, 0.0])]), Region(4.0)], weights=[1, 1])
        inter = AllOf([keep, left, right, PredicateSet(lambda q: True)])
        assert not is_finite(inter)
        got = seeds(inter)
        # [-1,0] fails `keep`; [1,0] from `left` is not in `right`... only configs the whole intersection contains
        assert all(inter.contains(m.q) for m in got)
        # Both [1,0] and [2,0] are seeds of some child, but neither is in every other child: intersection of
        # two different finite parts is empty here, so nothing is a member
        assert got == []

    def test_non_finite_intersection_keeps_seeds_the_intersection_contains(self):
        shared = np.array([1.0, 0.0])
        left = AnyOf([FiniteSet([shared, np.array([-1.0, 0.0])]), Region(4.0)], weights=[1, 1])
        right = AnyOf([FiniteSet([shared]), Region(4.0)], weights=[1, 1])
        inter = AllOf([PredicateSet(lambda q: True), left, right])  # first child bears no seeds
        assert not is_finite(inter)
        got = seeds(inter)
        assert [m.q.tolist() for m in got] == [[1.0, 0.0]]
        assert got[0].source == (0, 0)  # first occurrence wins: from `left`, its finite child, member 0

    def test_duplicate_seed_policy_is_first_occurrence(self):
        q = np.array([0.5, 0.5])
        a = AnyOf([FiniteSet([q]), Region(4.0)], weights=[1, 1])
        b = AnyOf([Region(4.0), FiniteSet([q])], weights=[1, 1])
        got = seeds(AllOf([b, a]))
        assert len(got) == 1 and got[0].source == (1, 0)  # from b: child 1, member 0

    @settings(max_examples=200, deadline=None)
    @given(st.data())
    def test_members_equal_membership_filter_of_all_leaf_configs(self, data):
        """When finite, members(s) is exactly the set of leaf configurations that s contains."""
        pool = [np.array([float(i), 0.0]) for i in range(4)]

        def finite():
            idx = data.draw(st.lists(st.integers(0, 3), min_size=1, max_size=3, unique=True))
            return FiniteSet([pool[i] for i in idx])

        def leaf():
            kind = data.draw(st.sampled_from(["finite", "pred", "sampler"]))
            if kind == "finite":
                return finite()
            if kind == "pred":
                c = data.draw(st.floats(-0.5, 3.5))
                return PredicateSet(lambda q, c=c: q[0] > c)
            return Deterministic([9.0, 9.0])

        def tree(depth):
            if depth == 0 or data.draw(st.booleans()):
                return leaf()
            kids = [tree(depth - 1) for _ in range(data.draw(st.integers(1, 3)))]
            if data.draw(st.booleans()):
                return AnyOf(
                    kids, weights=[1] * len(kids) if len(kids) > 1 and all(hasattr(k, "sample") for k in kids) else None
                )
            return AllOf(kids)

        s = tree(2)
        # Compare as sets of configurations: a union of two finite sets may list a shared
        # member once per child, with distinct provenance
        expected = sorted({tuple(q.tolist()) for q in pool if s.contains(q)})
        got = sorted({tuple(m.q.tolist()) for m in members(s)})
        if is_finite(s):
            assert got == expected
            assert sorted({tuple(m.q.tolist()) for m in seeds(s)}) == expected
        else:
            assert got == []
            # seeds are members, and every pool config that s contains and that some finite leaf lists
            # is found when it is embedded in a seed-bearing path
            assert all(s.contains(m.q) for m in seeds(s))
