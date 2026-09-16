# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for JointSpace."""

import numpy as np
import pytest

from pycbirrt.space import JointSpace


@pytest.fixture
def linear():
    return JointSpace(np.array([-1.0, -2.0]), np.array([1.0, 2.0]))


@pytest.fixture
def angular():
    return JointSpace(np.array([-np.pi, -1.0]), np.array([np.pi, 1.0]), angular_joints=(True, False))


class TestConstruction:
    def test_dof_and_limits(self, linear):
        assert linear.dof == 2
        lo, hi = linear.joint_limits
        assert np.array_equal(lo, [-1.0, -2.0])
        assert np.array_equal(hi, [1.0, 2.0])

    def test_angular_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="angular_joints length"):
            JointSpace(np.zeros(2), np.ones(2), angular_joints=(True, True, True))

    def test_bad_limits_raise(self):
        with pytest.raises(ValueError):
            JointSpace(np.zeros(2), np.ones(3))
        with pytest.raises(ValueError):
            JointSpace(np.ones(2), np.zeros(2))

    def test_all_false_angular_is_linear(self):
        s = JointSpace(np.zeros(2), np.ones(2), angular_joints=(False, False))
        assert s.angular_joints is None


class TestLimits:
    def test_within_limits_linear(self, linear):
        assert linear.within_limits(np.array([0.0, 0.0]))
        assert linear.within_limits(np.array([1.0, -2.0]))
        assert not linear.within_limits(np.array([1.1, 0.0]))
        assert not linear.within_limits(np.array([0.0, -2.1]))

    def test_angular_joint_ignores_limits(self, angular):
        assert angular.within_limits(np.array([100.0, 0.0]))
        assert not angular.within_limits(np.array([100.0, 1.5]))


class TestMetric:
    def test_linear_distance_is_euclidean(self, linear):
        assert linear.distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(5.0)

    def test_angular_distance_wraps(self, angular):
        q1 = np.array([-np.pi + 0.1, 0.0])
        q2 = np.array([np.pi - 0.1, 0.0])
        assert angular.distance(q1, q2) == pytest.approx(0.2)

    def test_angular_direction_short_way(self, angular):
        d = angular.direction(np.array([np.pi - 0.1, 0.0]), np.array([-np.pi + 0.1, 0.0]))
        assert d[0] == pytest.approx(0.2)

    def test_linear_joint_does_not_wrap(self, angular):
        d = angular.direction(np.array([0.0, -1.0]), np.array([0.0, 1.0]))
        assert d[1] == pytest.approx(2.0)

    def test_direction_does_not_mutate_inputs(self, angular):
        q_to = np.array([-np.pi + 0.1, 0.0])
        angular.direction(np.array([np.pi - 0.1, 0.0]), q_to)
        assert q_to[0] == pytest.approx(-np.pi + 0.1)


class TestInterpolate:
    def test_endpoints(self, linear):
        a, b = np.array([0.0, 0.0]), np.array([1.0, 2.0])
        assert np.allclose(linear.interpolate(a, b, 0.0), a)
        assert np.allclose(linear.interpolate(a, b, 1.0), b)
        assert np.allclose(linear.interpolate(a, b, 0.5), [0.5, 1.0])

    def test_angular_interpolation_crosses_pi(self, angular):
        a, b = np.array([np.pi - 0.1, 0.0]), np.array([-np.pi + 0.1, 0.0])
        mid = angular.interpolate(a, b, 0.5)
        # Halfway along the short path lands on pi (unwrapped)
        assert mid[0] == pytest.approx(np.pi)


class TestSample:
    def test_sample_within_limits(self, linear):
        rng = np.random.default_rng(0)
        for _ in range(100):
            assert linear.within_limits(linear.sample(rng))

    def test_sample_is_seedable(self, linear):
        a = linear.sample(np.random.default_rng(3))
        b = linear.sample(np.random.default_rng(3))
        assert np.array_equal(a, b)
