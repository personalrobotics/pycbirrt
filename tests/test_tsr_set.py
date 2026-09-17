# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for TSRConfigurationSet, the TSR-induced configuration set."""

import numpy as np
import pytest
from tsr import TSR

from pycbirrt.sets import AllOf, AnyOf, SetDistance, SetProjector, SetSampler, StateSet, supports
from pycbirrt.space import JointSpace
from pycbirrt.tsr_set import TSRConfigurationSet, tsr_weights
from tests.test_planner import MockCollisionChecker, MockIKSolver, MockRobotModel

BOX = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])


def frame(x=0.0, y=0.0):
    T = np.eye(4)
    T[0, 3], T[1, 3] = x, y
    return T


@pytest.fixture
def arm():
    robot = MockRobotModel()
    ik = MockIKSolver(robot, MockCollisionChecker())
    space = JointSpace(*robot.joint_limits)
    return robot, ik, space


def make_set(arm, tsr, **kw):
    robot, ik, space = arm
    return TSRConfigurationSet(tsr, robot, ik, space, **kw)


class TestCapabilities:
    def test_supports_all_three(self, arm):
        s = make_set(arm, TSR(T0_w=frame(1.2, 0.8), Tw_e=np.eye(4), Bw=BOX))
        assert supports(s, StateSet)
        assert supports(s, SetSampler)
        assert supports(s, SetDistance)
        assert supports(s, SetProjector)

    def test_bad_max_solutions(self, arm):
        with pytest.raises(ValueError):
            make_set(arm, TSR(Bw=BOX), max_solutions_per_pose=0)


class TestMembership:
    def test_contains_and_distance(self, arm):
        # q=[0,0] puts the EE at (2, 0); q=[pi/2, 0] at (0, 2)
        s = make_set(arm, TSR(T0_w=frame(2.0, 0.0), Tw_e=np.eye(4), Bw=BOX))
        assert s.contains(np.array([0.0, 0.0]))
        assert s.distance(np.array([0.0, 0.0])) == 0.0
        assert not s.contains(np.array([np.pi / 2, 0.0]))
        assert s.distance(np.array([np.pi / 2, 0.0])) > 1.0

    def test_matches_definition(self, arm):
        """contains(q) iff the TSR distance of FK(q) is within tolerance."""
        robot, _, _ = arm
        tsr = TSR(T0_w=frame(1.2, 0.8), Tw_e=np.eye(4), Bw=BOX)
        s = make_set(arm, tsr)
        rng = np.random.default_rng(0)
        for _ in range(200):
            q = rng.uniform(-np.pi, np.pi, 2)
            dist, _ = tsr.distance(robot.forward_kinematics(q))
            assert s.distance(q) == dist
            assert s.contains(q) == (dist <= s.tolerance)


class TestSampling:
    def test_samples_are_members_within_limits(self, arm):
        robot, ik, space = arm
        s = make_set(arm, TSR(T0_w=frame(1.2, 0.8), Tw_e=np.eye(4), Bw=BOX))
        rng = np.random.default_rng(1)
        n = 0
        for _ in range(50):
            smp = s.sample(rng)
            if smp is None:
                continue
            n += 1
            assert smp.source == ()
            assert space.within_limits(smp.q)
            assert s.contains(smp.q)
        assert n > 40

    def test_sampling_is_seedable(self, arm):
        s = make_set(arm, TSR(T0_w=frame(1.2, 0.8), Tw_e=np.eye(4), Bw=BOX))
        a = s.sample(np.random.default_rng(7))
        b = s.sample(np.random.default_rng(7))
        assert np.array_equal(a.q, b.q)

    def test_unreachable_returns_none(self, arm):
        s = make_set(arm, TSR(T0_w=frame(10.0, 0.0), Tw_e=np.eye(4), Bw=BOX))
        assert s.sample(np.random.default_rng(0)) is None

    def test_max_solutions_per_pose_reuses_pose(self, arm):
        robot, _, _ = arm
        s = make_set(arm, TSR(T0_w=frame(1.2, 0.8), Tw_e=np.eye(4), Bw=BOX), max_solutions_per_pose=2)
        rng = np.random.default_rng(3)
        a, b, c = s.sample(rng), s.sample(rng), s.sample(rng)
        # a and b are the two elbow solutions of one pose; c comes from a new pose
        pa, pb, pc = (robot.forward_kinematics(x.q)[:2, 3] for x in (a, b, c))
        assert np.allclose(pa, pb)
        assert not np.allclose(pa, pc)
        assert not np.allclose(a.q, b.q)

    def test_default_one_solution_per_pose(self, arm):
        robot, _, _ = arm
        s = make_set(arm, TSR(T0_w=frame(1.2, 0.8), Tw_e=np.eye(4), Bw=BOX))
        rng = np.random.default_rng(3)
        a, b = s.sample(rng), s.sample(rng)
        assert not np.allclose(robot.forward_kinematics(a.q)[:2, 3], robot.forward_kinematics(b.q)[:2, 3])

    def test_sample_pose_respects_frames(self, arm):
        Tw_e = frame(0.1, 0.0)
        tsr = TSR(T0_w=frame(1.2, 0.8), Tw_e=Tw_e, Bw=BOX)
        s = make_set(arm, tsr)
        rng = np.random.default_rng(0)
        for _ in range(20):
            pose = s.sample_pose(rng)
            assert tsr.contains(pose)


class TestProjection:
    @pytest.mark.parametrize(
        "tsr",
        [
            TSR(T0_w=frame(1.2, 0.8), Tw_e=np.eye(4), Bw=BOX),
            TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=BOX + np.array([[1.2], [0.8], [0], [0], [0], [0]])),
            TSR(T0_w=frame(1.2, 0.8), Tw_e=frame(0.1, 0.0), Bw=BOX),
        ],
        ids=["T0_w", "identity", "T0_w+Tw_e"],
    )
    def test_projection_lands_in_set(self, arm, tsr):
        s = make_set(arm, tsr)
        q = np.array([0.3, 0.9])
        assert not s.contains(q)
        q_proj = s.project(q, q)
        assert q_proj is not None
        assert s.contains(q_proj)

    def test_projection_is_identity_inside(self, arm):
        s = make_set(arm, TSR(T0_w=frame(2.0, 0.0), Tw_e=np.eye(4), Bw=BOX))
        q = np.array([0.0, 0.0])
        assert np.array_equal(s.project(q, q), q)

    def test_projection_returns_none_when_unreachable(self, arm):
        s = make_set(arm, TSR(T0_w=frame(10.0, 0.0), Tw_e=np.eye(4), Bw=BOX))
        assert s.project(np.zeros(2), np.zeros(2)) is None


class TestComposition:
    def test_anyof_with_volume_weights(self, arm):
        a = make_set(arm, TSR(T0_w=frame(1.2, 0.8), Tw_e=np.eye(4), Bw=BOX))
        b = make_set(arm, TSR(T0_w=frame(-1.2, 0.8), Tw_e=np.eye(4), Bw=np.zeros((6, 2))))
        w = tsr_weights([a, b])
        assert w[0] > w[1] == 0.0
        union = AnyOf([a, b], weights=w)
        assert supports(union, SetSampler)
        rng = np.random.default_rng(0)
        for _ in range(20):
            smp = union.sample(rng)
            if smp is not None:
                assert smp.source == (0,)
                assert union.contains(smp.q)

    def test_zero_volume_fallback_is_uniform(self, arm):
        a = make_set(arm, TSR(Bw=np.zeros((6, 2))))
        b = make_set(arm, TSR(Bw=np.zeros((6, 2))))
        assert np.array_equal(tsr_weights([a, b]), [1.0, 1.0])

    def test_allof_single_child_projects(self, arm):
        s = make_set(arm, TSR(T0_w=frame(1.2, 0.8), Tw_e=np.eye(4), Bw=BOX))
        path_set = AllOf([s])
        assert supports(path_set, SetProjector)
        q = np.array([0.3, 0.9])
        assert path_set.contains(path_set.project(q, q))
