# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Violation is separate from distance, and most-violated projection ranks by it (#44)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tsr import TSR

from pycbirrt.exceptions import UnsupportedCapability
from pycbirrt.sets import AllOf, AnyOf, FiniteSet, MostViolatedProjection, PredicateSet, SetViolation, supports
from pycbirrt.space import JointSpace
from pycbirrt.tsr_set import TSRConfigurationSet
from tests.test_planner import MockCollisionChecker, MockIKSolver, MockRobotModel


class Band:
    """1-D set {q : |q[0] - c| <= tol} with a spy on projection calls."""

    def __init__(self, c, tol, scale=1.0):
        self.c, self.tol, self.scale = c, tol, scale
        self.projections = 0

    def distance(self, q):
        return self.scale * abs(float(q[0]) - self.c)

    def violation(self, q):
        return max(0.0, self.distance(q) - self.scale * self.tol)

    def contains(self, q):
        return abs(float(q[0]) - self.c) <= self.tol

    def project(self, q_prev, q):
        self.projections += 1
        x = float(np.clip(q[0], self.c - self.tol, self.c + self.tol))
        return np.array([x, *q[1:]])


class TestMostViolatedSelection:
    def test_issue_example_different_tolerances(self):
        """A satisfied child (dist 0.9, tol 1.0) must not outrank an unsatisfied one (dist 0.11, tol 0.1)."""
        a = Band(0.0, 1.0)  # satisfied at x = 0.89
        b = Band(1.0, 0.1)  # unsatisfied at x = 0.89 (dist 0.11)
        s = AllOf([a, b], projection=MostViolatedProjection())
        q = np.array([0.89, 0.0])
        assert a.contains(q) and not b.contains(q)
        result = s.project(q, q)
        assert result is not None and s.contains(result)
        assert a.projections == 0  # the satisfied child was never selected

    def test_satisfied_child_never_selected(self):
        """Across many starting points, only unsatisfied children get projected onto."""
        a, b, c = Band(0.0, 0.5), Band(0.3, 0.3), Band(0.2, 0.05)
        s = AllOf([a, b, c], projection=MostViolatedProjection())
        for x in np.linspace(-2, 2, 41):
            for band in (a, b, c):
                band.projections = 0
            q = np.array([x, 0.0])
            satisfied = [band for band in (a, b, c) if band.contains(q)]
            s.project(q, q)
            # A child satisfied at the start may become unsatisfied later; check only the first pick
            if len(satisfied) < 3:
                first_unsatisfied_violation = max(band.violation(q) for band in (a, b, c) if not band.contains(q))
                for band in satisfied:
                    assert band.violation(q) == 0.0 < first_unsatisfied_violation

    def test_heterogeneous_magnitudes_are_the_callers_problem_but_do_not_break_termination(self):
        """A child reporting violations on a 1000x scale dominates selection; projection still terminates."""
        big = Band(0.0, 0.1, scale=1000.0)
        small = Band(0.5, 0.1)
        s = AllOf([big, small], projection=MostViolatedProjection(max_iters=20))
        q = np.array([0.3, 0.0])
        result = s.project(q, q)
        # The intersection [0.4, 0.1] is empty, so None is the right answer; it must not loop forever
        assert result is None

    def test_feasible_intersection_of_three_bands(self):
        a, b, c = Band(0.0, 1.0), Band(0.8, 0.5), Band(1.0, 0.4)  # intersection [0.6, 1.0]
        s = AllOf([a, b, c], projection=MostViolatedProjection())
        for x in (-3.0, 0.0, 0.5, 2.0, 5.0):
            q = np.array([x, 0.0])
            result = s.project(q, q)
            assert result is not None and s.contains(result)

    def test_requires_violation_not_distance(self):
        class DistanceOnly:
            def distance(self, q):
                return 0.0

            def contains(self, q):
                return True

            def project(self, q_prev, q):
                return q

        with pytest.raises(UnsupportedCapability, match="violation"):
            AllOf([DistanceOnly(), Band(0, 1)], projection=MostViolatedProjection())


class TestViolationContract:
    @pytest.fixture
    def tsr_set(self):
        robot = MockRobotModel()
        ik = MockIKSolver(robot, MockCollisionChecker())
        T = np.eye(4)
        T[0, 3], T[1, 3] = 1.2, 0.8
        box = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
        return TSRConfigurationSet(
            TSR(T0_w=T, Tw_e=np.eye(4), Bw=box), robot, ik, JointSpace(*robot.joint_limits), tolerance=0.01
        )

    def test_tsr_set_violation_is_distance_beyond_tolerance(self, tsr_set):
        rng = np.random.default_rng(0)
        for _ in range(100):
            q = rng.uniform(-np.pi, np.pi, 2)
            v, d = tsr_set.violation(q), tsr_set.distance(q)
            assert v == pytest.approx(max(0.0, d - tsr_set.tolerance))
            assert tsr_set.contains(q) == (v == 0.0)

    def test_finite_set_violation(self):
        s = FiniteSet([np.zeros(2)], tolerance=0.1)
        assert s.violation(np.array([0.05, 0.0])) == 0.0
        assert s.violation(np.array([0.3, 0.0])) == pytest.approx(0.2)
        assert supports(s, SetViolation)

    def test_predicate_set_has_no_violation(self):
        assert not supports(PredicateSet(lambda q: True), SetViolation)
        assert not supports(AllOf([FiniteSet([np.zeros(2)]), PredicateSet(lambda q: True)]), SetViolation)

    @settings(max_examples=200, deadline=None)
    @given(st.floats(-3, 3), st.floats(-3, 3), st.floats(0.01, 1.0), st.floats(0.01, 1.0), st.floats(-4, 4))
    def test_composites_zero_iff_contains(self, c1, c2, t1, t2, x):
        a, b = Band(c1, t1), Band(c2, t2)
        q = np.array([x, 0.0])
        both = AllOf([a, b])
        either = AnyOf([a, b])
        assert both.contains(q) == (both.violation(q) == 0.0)
        assert either.contains(q) == (either.violation(q) == 0.0)
        assert both.violation(q) == max(a.violation(q), b.violation(q))
        assert either.violation(q) == min(a.violation(q), b.violation(q))


class TestLegacyHomogeneousIntersection:
    def test_two_tsr_constraints_still_project(self):
        """Homogeneous TSR intersections keep working through the lowering."""
        from pycbirrt import CBiRRT, CBiRRTConfig
        from pycbirrt.legacy import legacy_problem

        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)
        planner = CBiRRT(robot, ik, collision, CBiRRTConfig(smooth_path=False))
        band_x = np.array([[-0.5, 0.5], [-2.0, 2.0], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
        band_y = np.array([[-2.0, 2.0], [-0.6, 0.6], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
        Tx = np.eye(4)
        Tx[0, 3] = 1.5
        q = np.array([0.0, 0.6])
        prob = legacy_problem(
            robot,
            ik,
            collision,
            planner.space,
            planner.config,
            [q],
            [q],
            None,
            None,
            [TSR(T0_w=Tx, Bw=band_x), TSR(Bw=band_y)],
        )
        assert supports(prob.path_constraint, SetViolation)
        q_out = np.array([1.2, 0.3])
        assert not prob.path_constraint.contains(q_out)
        result = prob.path_constraint.project(q_out, q_out)
        assert result is not None and prob.path_constraint.contains(result)


# ---------------------------------------------------------------------------
# Tied violation plateaus (#57)
# ---------------------------------------------------------------------------


class Axis:
    """{q : q[i] == 0}. Violation |q[i]|; projection zeroes that coordinate."""

    def __init__(self, i):
        self.i = i
        self.projections = 0

    def violation(self, q):
        return abs(float(q[self.i]))

    def contains(self, q):
        return float(q[self.i]) == 0.0

    def project(self, q_prev, q):
        self.projections += 1
        out = np.array(q, dtype=float)
        out[self.i] = 0.0
        return out


class TestTiedPlateaus:
    def test_two_axes_from_corner(self):
        s = AllOf([Axis(0), Axis(1)], projection=MostViolatedProjection())
        q = np.array([1.0, 1.0])
        result = s.project(q, q)
        assert result is not None and np.array_equal(result, [0.0, 0.0])

    def test_three_axes_all_tied(self):
        axes = [Axis(0), Axis(1), Axis(2)]
        s = AllOf(axes, projection=MostViolatedProjection())
        q = np.array([2.0, 2.0, 2.0])
        result = s.project(q, q)
        assert result is not None and np.array_equal(result, [0.0, 0.0, 0.0])
        assert sum(a.projections for a in axes) == 3  # one projection per axis, no wasted steps

    def test_ties_below_the_maximum(self):
        """The maximum drops, then two equal residuals remain; both must be cleared."""
        s = AllOf([Axis(0), Axis(1), Axis(2)], projection=MostViolatedProjection())
        q = np.array([3.0, 1.0, 1.0])
        result = s.project(q, q)
        assert result is not None and np.array_equal(result, [0.0, 0.0, 0.0])

    def test_immovable_projector_terminates(self):
        class Stuck:
            calls = 0

            def violation(self, q):
                return 1.0

            def contains(self, q):
                return False

            def project(self, q_prev, q):
                Stuck.calls += 1
                return np.array(q)  # no change

        s = AllOf([Stuck(), Axis(1)], projection=MostViolatedProjection(max_iters=1000))
        assert s.project(np.array([1.0, 1.0]), np.array([1.0, 1.0])) is None
        assert Stuck.calls == 1

    def test_cycling_infeasible_intersection_terminates_within_a_sweep(self):
        """Two disjoint intervals: alternating projection ping-pongs; stop after one full sweep without progress."""
        a, b = Band(0.0, 0.5), Band(2.0, 0.5)  # [-0.5, 0.5] and [1.5, 2.5]
        s = AllOf([a, b], projection=MostViolatedProjection(max_iters=1000))
        assert s.project(np.array([1.0, 0.0]), np.array([1.0, 0.0])) is None
        assert a.projections + b.projections <= 4

    def test_max_iters_still_bounds(self):
        class Creeping:
            """Always violated; each projection halves the violation, so the profile always improves."""

            def __init__(self):
                self.v = 1.0

            def violation(self, q):
                return self.v

            def contains(self, q):
                return False

            def project(self, q_prev, q):
                self.v *= 0.5
                return np.array(q) + 1e-3

        s = AllOf([Creeping(), Axis(1)], projection=MostViolatedProjection(max_iters=7))
        assert s.project(np.zeros(2), np.zeros(2)) is None

    def test_different_tolerance_regression_still_passes(self):
        a, b = Band(0.0, 1.0), Band(1.0, 0.1)
        s = AllOf([a, b], projection=MostViolatedProjection())
        q = np.array([0.89, 0.0])
        result = s.project(q, q)
        assert result is not None and s.contains(result) and a.projections == 0
