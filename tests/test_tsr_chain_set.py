# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""A TSR chain is one pose region and induces one configuration set (#7).

The planar arm swings a door: a hinge TSR at (1.0, 0.5) with free yaw, then a
handle TSR fixed 0.3 along the door's x axis, grasped from any angle. The
chain's poses lie on a circle of radius 0.3 around the hinge, which no single
TSR and no intersection of the two components describes.
"""

import numpy as np
import pytest
from tsr import TSR, TSRChain

from sscbirrt import CBiRRT, CBiRRTConfig, PlanningProblem
from sscbirrt.exceptions import UnsupportedCapability
from sscbirrt.legacy import legacy_problem
from sscbirrt.sets import AllOf, AnyOf, FiniteSet, SetDistance, SetProjector, SetSampler, SetViolation, supports
from sscbirrt.space import JointSpace
from sscbirrt.tsr_set import PoseRegion, TSRConfigurationSet, region_volume, tsr_weights
from tests.test_planner import MockCollisionChecker, MockIKSolver, MockRobotModel

HINGE = np.array([1.0, 0.5])
FIXED = [[0, 0]] * 5


def frame(x, y):
    T = np.eye(4)
    T[0, 3], T[1, 3] = x, y
    return T


def hinge_tsr(yaw=(-np.pi, np.pi)):
    return TSR(T0_w=frame(*HINGE), Tw_e=np.eye(4), Bw=np.array(FIXED + [list(yaw)]))


def handle_tsr(x=(0.3, 0.3)):
    # Free yaw: the planar mock's FK carries no orientation, and a handle may be grasped from any angle
    return TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=np.array([list(x)] + FIXED[:4] + [[-np.pi, np.pi]]))


def door_chain(yaw=(-np.pi, np.pi), x=(0.3, 0.3)):
    return TSRChain(TSRs=[hinge_tsr(yaw), handle_tsr(x)])


@pytest.fixture
def arm():
    robot = MockRobotModel()
    collision = MockCollisionChecker()
    ik = MockIKSolver(robot, collision)
    planner = CBiRRT(robot, ik, collision, CBiRRTConfig(smooth_path=False))
    return robot, ik, collision, planner


def chain_set(arm, chain, **kw):
    robot, ik, _, planner = arm
    return TSRConfigurationSet(chain, robot, ik, planner.space, **kw)


def radius_from_hinge(robot, q):
    return float(np.linalg.norm(robot.forward_kinematics(q)[:2, 3] - HINGE))


class TestChainAsRegion:
    def test_chain_and_tsr_are_pose_regions(self):
        assert isinstance(door_chain(), PoseRegion)
        assert isinstance(hinge_tsr(), PoseRegion)
        assert not isinstance(object(), PoseRegion)

    def test_adapter_rejects_non_regions(self, arm):
        with pytest.raises(TypeError, match="pose region"):
            chain_set(arm, object())

    def test_capabilities(self, arm):
        s = chain_set(arm, door_chain())
        for cap in (SetSampler, SetDistance, SetViolation, SetProjector):
            assert supports(s, cap)

    def test_samples_lie_on_the_circle_and_are_members(self, arm):
        robot, _, _, planner = arm
        s = chain_set(arm, door_chain())
        rng = np.random.default_rng(0)
        draws = 0
        for _ in range(20):
            cands = s.sample(rng)
            if not cands:
                continue
            draws += 1
            for c in cands:
                assert planner.space.within_limits(c.q)
                assert radius_from_hinge(robot, c.q) == pytest.approx(0.3, abs=1e-6)
                assert s.contains(c.q)
        assert draws >= 15

    def test_sampling_is_seedable(self, arm):
        s = chain_set(arm, door_chain())
        a, b = s.sample(np.random.default_rng(4)), s.sample(np.random.default_rng(4))
        assert len(a) == len(b) and all(np.array_equal(x.q, y.q) for x, y in zip(a, b))

    def test_projection_lands_on_the_circle(self, arm):
        robot, _, _, _ = arm
        s = chain_set(arm, door_chain())
        q = np.array([0.6, 0.4])  # EE well off the circle
        assert not s.contains(q)
        q_proj = s.project(q, q)
        assert q_proj is not None and s.contains(q_proj)
        assert radius_from_hinge(robot, q_proj) == pytest.approx(0.3, abs=1e-3)

    def test_violation_zero_iff_contains(self, arm):
        s = chain_set(arm, door_chain())
        rng = np.random.default_rng(1)
        for _ in range(30):
            q = rng.uniform(-np.pi, np.pi, 2)
            assert s.contains(q) == (s.violation(q) == 0.0)

    def test_chain_is_not_the_intersection_of_its_components(self, arm):
        """A chain member is generally in neither component's world-frame set."""
        robot, _, _, _ = arm
        chain = chain_set(arm, door_chain())
        hinge = chain_set(arm, hinge_tsr())  # poses exactly at the hinge, any yaw
        handle = chain_set(arm, handle_tsr())  # poses exactly at world (0.3, 0)
        both = AllOf([hinge, handle])
        rng = np.random.default_rng(2)
        member = next(c.q for _ in range(20) for c in chain.sample(rng))
        assert chain.contains(member)
        assert not hinge.contains(member) and not handle.contains(member) and not both.contains(member)

    def test_region_volume_and_weights(self, arm):
        c = chain_set(arm, door_chain())
        t = chain_set(arm, hinge_tsr(yaw=(0, 0)))
        assert region_volume(door_chain()) == pytest.approx(4 * np.pi)  # hinge yaw + handle yaw
        w = tsr_weights([c, t])
        assert w[0] == pytest.approx(4 * np.pi) and w[1] == 0.0


class TestPlanningWithChains:
    def test_solve_to_a_chain_goal(self, arm):
        robot, _, collision, planner = arm
        goal = chain_set(arm, door_chain())
        problem = PlanningProblem(space=planner.space, start=FiniteSet([np.zeros(2)]), goal=goal, validator=collision)
        result = planner.solve(problem, seed=0)
        assert result.success
        assert goal.contains(result.path[-1])
        assert radius_from_hinge(robot, result.path[-1]) == pytest.approx(0.3, abs=1e-3)

    def test_legacy_plan_accepts_a_chain_goal(self, arm):
        robot, _, _, planner = arm
        result = planner.plan(start=np.zeros(2), goal_tsrs=[door_chain()], seed=0, return_details=True)
        assert result.success and result.goal_index == 0
        assert door_chain().contains(robot.forward_kinematics(result.path[-1]))

    def test_union_of_chain_and_tsr_reports_which(self, arm):
        robot, _, _, planner = arm
        near_point = TSR(
            T0_w=frame(1.9, 0.3),
            Tw_e=np.eye(4),
            Bw=np.array([[-0.05, 0.05], [-0.05, 0.05]] + FIXED[:3] + [[-np.pi, np.pi]]),
        )
        result = planner.plan(start=np.zeros(2), goal_tsrs=[door_chain(), near_point], seed=3, return_details=True)
        assert result.success
        reached = [door_chain(), near_point][result.goal_index]
        assert reached.contains(robot.forward_kinematics(result.path[-1]))

    def test_chain_as_path_constraint(self, arm):
        """An annulus around the hinge (handle offset in [0.2, 0.4]) constrains the whole path."""
        robot, ik, collision, _ = arm
        planner = CBiRRT(robot, ik, collision, CBiRRTConfig(smooth_path=False, step_size=0.2, timeout=60.0))
        annulus = door_chain(x=(0.2, 0.4))
        constraint = chain_set(arm, annulus)
        # Start and goal on the annulus, on opposite sides of the hinge
        starts = [c.q for c in constraint.sample(np.random.default_rng(5))]
        goals = [c.q for c in constraint.sample(np.random.default_rng(6))]
        assert starts and goals
        problem = PlanningProblem(
            space=planner.space,
            start=FiniteSet([starts[0]]),
            goal=FiniteSet([goals[0]]),
            validator=collision,
            path_constraint=constraint,
        )
        result = planner.solve(problem, seed=0)
        assert result.success, result.failure_reason
        for q in result.path:
            assert 0.2 - 2e-3 <= radius_from_hinge(robot, q) <= 0.4 + 2e-3
            assert constraint.contains(q)

    def test_legacy_plan_accepts_a_chain_constraint(self, arm):
        robot, ik, collision, _ = arm
        planner = CBiRRT(robot, ik, collision, CBiRRTConfig(smooth_path=False, step_size=0.2, timeout=60.0))
        annulus = door_chain(x=(0.2, 0.4))
        s = chain_set(arm, annulus)
        q0 = next(c.q for c in s.sample(np.random.default_rng(7)))
        q1 = next(c.q for c in s.sample(np.random.default_rng(8)))
        path = planner.plan(start=q0, goal=q1, constraint_tsrs=[annulus], seed=1)
        assert path is not None
        assert all(annulus.contains(robot.forward_kinematics(q)) for q in path)

    def test_lowering_builds_one_set_per_chain(self, arm):
        robot, ik, collision, planner = arm
        problem = legacy_problem(
            robot, ik, collision, planner.space, planner.config, [np.zeros(2)], None, None, [door_chain()], None
        )
        assert isinstance(problem.goal, AnyOf) and len(problem.goal.children) == 1
        assert isinstance(problem.goal.children[0].tsr, TSRChain)

    def test_membership_only_chain_set_is_not_needed(self, arm):
        """Sanity: a chain set can seed a goal, so it never hits the unsupported-capability path."""
        _, _, collision, planner = arm
        goal = chain_set(arm, door_chain())
        try:
            planner.solve(
                PlanningProblem(space=planner.space, start=FiniteSet([np.zeros(2)]), goal=goal, validator=collision),
                seed=0,
            )
        except UnsupportedCapability:  # pragma: no cover
            pytest.fail("chain set should be sampleable")


class TestJointSpaceStillGoverns:
    def test_out_of_limit_ik_branches_are_dropped(self, arm):
        robot, ik, _, _ = arm
        space = JointSpace(np.array([-np.pi, 0.0]), np.array([np.pi, np.pi]))  # elbow non-negative only
        s = TSRConfigurationSet(door_chain(), robot, ik, space)
        rng = np.random.default_rng(0)
        for _ in range(20):
            for c in s.sample(rng):
                assert space.within_limits(c.q)
