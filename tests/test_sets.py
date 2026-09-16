# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for state sets: membership, composition, and capabilities."""

import numpy as np
import pytest

from pycbirrt.exceptions import UnsupportedCapability
from pycbirrt.sets import (
    AllOf,
    AnyOf,
    FiniteSet,
    MostViolatedProjection,
    PredicateSet,
    Sample,
    SetDistance,
    SetProjector,
    SetSampler,
    StateSet,
    supports,
)


class Halfspace:
    """1-D test set: {q : q[0] >= lo} with distance, sampling, and projection."""

    def __init__(self, lo: float, hi: float = 10.0, tolerance: float = 1e-9):
        self.lo = lo
        self.hi = hi  # only used to bound sampling
        self.tolerance = tolerance

    def distance(self, q):
        return max(0.0, self.lo - float(q[0]))

    def contains(self, q):
        return self.distance(q) <= self.tolerance

    def sample(self, rng):
        return Sample(np.array([rng.uniform(self.lo, self.hi)]))

    def project(self, q_previous, q_proposed):
        return np.array([max(self.lo, float(q_proposed[0]))])


class Interval:
    """1-D test set: {q : lo <= q[0] <= hi} with distance and projection, no sampler."""

    def __init__(self, lo: float, hi: float, tolerance: float = 1e-9):
        self.lo, self.hi, self.tolerance = lo, hi, tolerance

    def distance(self, q):
        x = float(q[0])
        return max(0.0, self.lo - x, x - self.hi)

    def contains(self, q):
        return self.distance(q) <= self.tolerance

    def project(self, q_previous, q_proposed):
        return np.array([float(np.clip(q_proposed[0], self.lo, self.hi))])


class Membership:
    """Membership-only set."""

    def __init__(self, ok):
        self.ok = ok

    def contains(self, q):
        return self.ok


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------


class TestFiniteSet:
    def test_contains_within_tolerance(self):
        s = FiniteSet([np.array([0.0, 0.0]), np.array([1.0, 1.0])], tolerance=1e-3)
        assert s.contains(np.array([0.0, 0.0]))
        assert s.contains(np.array([1.0, 1.0005]))
        assert not s.contains(np.array([0.5, 0.5]))

    def test_distance_to_nearest_member(self):
        s = FiniteSet([np.array([0.0]), np.array([2.0])])
        assert s.distance(np.array([1.5])) == pytest.approx(0.5)

    def test_sample_reports_member_index(self):
        configs = [np.array([0.0]), np.array([1.0]), np.array([2.0])]
        s = FiniteSet(configs)
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(50):
            smp = s.sample(rng)
            assert smp is not None
            (i,) = smp.source
            assert np.array_equal(smp.q, configs[i])
            seen.add(i)
        assert seen == {0, 1, 2}

    def test_custom_metric(self):
        def wrap(a, b):
            d = np.arctan2(np.sin(b - a), np.cos(b - a))
            return float(np.linalg.norm(d))

        s = FiniteSet([np.array([np.pi])], tolerance=1e-6, metric=wrap)
        assert s.contains(np.array([-np.pi]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            FiniteSet([])

    def test_capabilities(self):
        s = FiniteSet([np.array([0.0])])
        assert supports(s, StateSet)
        assert supports(s, SetSampler)
        assert supports(s, SetDistance)
        assert not supports(s, SetProjector)


class TestPredicateSet:
    def test_membership_only(self):
        s = PredicateSet(lambda q: q[0] > 0, name="positive")
        assert s.contains(np.array([1.0]))
        assert not s.contains(np.array([-1.0]))
        assert supports(s, StateSet)
        assert not supports(s, SetSampler)
        assert not supports(s, SetDistance)
        assert not supports(s, SetProjector)


# ---------------------------------------------------------------------------
# Union
# ---------------------------------------------------------------------------


class TestAnyOf:
    def test_membership_is_union(self):
        s = AnyOf([Interval(0, 1), Interval(2, 3)])
        assert s.contains(np.array([0.5]))
        assert s.contains(np.array([2.5]))
        assert not s.contains(np.array([1.5]))

    def test_distance_is_min(self):
        s = AnyOf([Interval(0, 1), Interval(2, 3)])
        assert s.distance(np.array([1.2])) == pytest.approx(0.2)
        assert s.distance(np.array([1.9])) == pytest.approx(0.1)

    def test_distance_unsupported_if_any_child_lacks_it(self):
        s = AnyOf([Interval(0, 1), Membership(True)])
        assert not supports(s, SetDistance)
        with pytest.raises(UnsupportedCapability):
            s.distance(np.array([0.0]))

    def test_single_child_delegates_everything(self):
        child = Halfspace(1.0)
        s = AnyOf([child])
        assert supports(s, SetSampler)
        assert supports(s, SetProjector)
        assert supports(s, SetDistance)
        rng = np.random.default_rng(0)
        smp = s.sample(rng)
        assert smp is not None and child.contains(smp.q)
        assert smp.source == (0,)
        assert s.project(np.array([0.0]), np.array([-1.0]))[0] == pytest.approx(1.0)

    def test_multi_child_sampling_requires_weights(self):
        s = AnyOf([Halfspace(0.0), Halfspace(5.0)])
        assert not supports(s, SetSampler)
        with pytest.raises(UnsupportedCapability, match="mixture policy"):
            s.sample(np.random.default_rng(0))

    def test_weighted_sampling_reports_child_index(self):
        a, b = Halfspace(0.0, 1.0), Halfspace(5.0, 6.0)
        s = AnyOf([a, b], weights=[1.0, 3.0])
        rng = np.random.default_rng(0)
        counts = [0, 0]
        for _ in range(400):
            smp = s.sample(rng)
            assert smp is not None
            i = smp.source[0]
            assert [a, b][i].contains(smp.q)
            counts[i] += 1
        # Roughly 1:3 mixture
        assert counts[1] > 2 * counts[0]

    def test_zero_weight_child_never_chosen(self):
        s = AnyOf([Halfspace(0.0, 1.0), Halfspace(5.0, 6.0)], weights=[1.0, 0.0])
        rng = np.random.default_rng(0)
        for _ in range(50):
            assert s.sample(rng).source[0] == 0

    def test_bad_weights_raise(self):
        with pytest.raises(ValueError):
            AnyOf([Halfspace(0.0), Halfspace(1.0)], weights=[1.0])
        with pytest.raises(ValueError):
            AnyOf([Halfspace(0.0), Halfspace(1.0)], weights=[0.0, 0.0])
        with pytest.raises(ValueError):
            AnyOf([Halfspace(0.0), Halfspace(1.0)], weights=[-1.0, 1.0])

    def test_child_sample_failure_propagates(self):
        class Fails:
            def contains(self, q):
                return False

            def sample(self, rng):
                return None

        s = AnyOf([Fails()])
        assert s.sample(np.random.default_rng(0)) is None

    def test_projection_picks_nearest_child_result(self):
        s = AnyOf([Interval(0, 1), Interval(4, 5)])
        assert supports(s, SetProjector)
        assert s.project(np.array([0.0]), np.array([1.5]))[0] == pytest.approx(1.0)
        assert s.project(np.array([0.0]), np.array([3.9]))[0] == pytest.approx(4.0)

    def test_projection_unsupported_if_any_child_lacks_it(self):
        s = AnyOf([Interval(0, 1), Membership(True)])
        assert not supports(s, SetProjector)
        with pytest.raises(UnsupportedCapability):
            s.project(np.array([0.0]), np.array([0.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            AnyOf([])


# ---------------------------------------------------------------------------
# Intersection
# ---------------------------------------------------------------------------


class TestAllOf:
    def test_membership_is_intersection(self):
        s = AllOf([Interval(0, 2), Interval(1, 3)])
        assert s.contains(np.array([1.5]))
        assert not s.contains(np.array([0.5]))
        assert not s.contains(np.array([2.5]))

    def test_distance_is_max_violation(self):
        s = AllOf([Interval(0, 2), Interval(1, 3)])
        assert s.distance(np.array([0.0])) == pytest.approx(1.0)
        assert s.distance(np.array([1.5])) == 0.0

    def test_single_child_delegates_everything(self):
        child = Halfspace(1.0)
        s = AllOf([child])
        assert supports(s, SetSampler)
        assert supports(s, SetProjector)
        smp = s.sample(np.random.default_rng(0))
        assert smp is not None and child.contains(smp.q)
        assert smp.source == ()
        assert s.project(np.array([0.0]), np.array([-1.0]))[0] == pytest.approx(1.0)

    def test_multi_child_projection_requires_strategy(self):
        s = AllOf([Interval(0, 2), Interval(1, 3)])
        assert not supports(s, SetProjector)
        with pytest.raises(UnsupportedCapability, match="projection strategy"):
            s.project(np.array([0.0]), np.array([0.0]))

    def test_multi_child_sampling_unsupported(self):
        s = AllOf([Halfspace(0.0), Halfspace(1.0)])
        assert not supports(s, SetSampler)
        with pytest.raises(UnsupportedCapability):
            s.sample(np.random.default_rng(0))

    def test_most_violated_projection(self):
        s = AllOf([Interval(0, 2), Interval(1, 3)], projection=MostViolatedProjection())
        assert supports(s, SetProjector)
        q = s.project(np.array([0.0]), np.array([-5.0]))
        assert q is not None and s.contains(q)
        assert q[0] == pytest.approx(1.0)

    def test_most_violated_projection_returns_none_on_no_progress(self):
        # Disjoint intervals: alternating projection ping-pongs without converging
        s = AllOf([Interval(0, 1), Interval(2, 3)], projection=MostViolatedProjection(max_iters=20))
        assert s.project(np.array([0.0]), np.array([0.5])) is None

    def test_most_violated_requires_child_capabilities(self):
        with pytest.raises(UnsupportedCapability, match="child 1"):
            AllOf([Interval(0, 1), Membership(True)], projection=MostViolatedProjection())

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            AllOf([])


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


class TestNesting:
    def test_grouping_is_preserved(self):
        # (L1 & R1) | (L2 & R2) vs (L1 | L2) & (R1 | R2) on a 2-D state
        def box(i, lo, hi):
            return PredicateSet(lambda q, i=i: lo <= q[i] <= hi)

        L1, L2 = box(0, 0, 1), box(0, 2, 3)
        R1, R2 = box(1, 0, 1), box(1, 2, 3)
        matched = AnyOf([AllOf([L1, R1]), AllOf([L2, R2])])
        crossed = AllOf([AnyOf([L1, L2]), AnyOf([R1, R2])])

        q_matched = np.array([0.5, 0.5])
        q_crossed = np.array([0.5, 2.5])
        assert matched.contains(q_matched) and crossed.contains(q_matched)
        assert not matched.contains(q_crossed)
        assert crossed.contains(q_crossed)

    def test_nested_sample_provenance(self):
        inner = AnyOf([FiniteSet([np.array([0.0])]), FiniteSet([np.array([1.0])])], weights=[1, 1])
        outer = AnyOf([FiniteSet([np.array([9.0])]), inner], weights=[1, 1])
        rng = np.random.default_rng(1)
        for _ in range(30):
            smp = outer.sample(rng)
            if smp.source[0] == 0:
                assert smp.source == (0, 0)
                assert smp.q[0] == 9.0
            else:
                assert smp.source[:1] == (1,)
                assert smp.q[0] == float(smp.source[1])
                assert smp.source[2] == 0

    def test_composites_are_state_sets(self):
        assert supports(AnyOf([Membership(True)]), StateSet)
        assert supports(AllOf([Membership(True), Membership(False)]), StateSet)

    def test_nested_capabilities_propagate(self):
        leaf = AllOf([Interval(0, 2), Interval(1, 3)], projection=MostViolatedProjection())
        s = AnyOf([leaf, Interval(5, 6)])
        assert supports(s, SetProjector)
        assert supports(s, SetDistance)
        assert not supports(s, SetSampler)
        q = s.project(np.array([0.0]), np.array([-1.0]))
        assert q[0] == pytest.approx(1.0)
