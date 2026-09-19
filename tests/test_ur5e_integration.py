# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Integration check on a real arm: constrained planning with the UR5e in MuJoCo, SSIK IK.

Skipped unless mujoco and ssik are installed and MUJOCO_MENAGERIE_PATH points
at a clone of google-deepmind/mujoco_menagerie. This is the check that caught
the root-sampling regression in #33: the start region below has several IK
branches per pose of which only some are collision-free, and never the first.
With SSIK every in-limit winding is returned as well (#36, #63).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
ssik = pytest.importorskip("ssik")
MENAGERIE = os.environ.get("MUJOCO_MENAGERIE_PATH")
if not MENAGERIE or not (Path(MENAGERIE) / "universal_robots_ur5e").exists():
    pytest.skip("MUJOCO_MENAGERIE_PATH not set to a mujoco_menagerie clone", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from tsr import TSR  # noqa: E402
from ur5e_mujoco import create_grasp_tsr, create_scene  # noqa: E402

from pycbirrt import CBiRRT, CBiRRTConfig  # noqa: E402
from pycbirrt.backends.mujoco import MuJoCoCollisionChecker, MuJoCoRobotModel, site_offset_in_body  # noqa: E402
from pycbirrt.backends.ssik import SSIKSolver  # noqa: E402
from pycbirrt.space import JointSpace  # noqa: E402
from pycbirrt.tsr_set import TSRConfigurationSet  # noqa: E402

JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@pytest.fixture(scope="module")
def ur5e():
    model = create_scene(Path(MENAGERIE))
    data = mujoco.MjData(model)
    robot = MuJoCoRobotModel(model, data, "attachment_site", JOINTS)
    collision = MuJoCoCollisionChecker(model, data, JOINTS)
    arm = ssik.Manipulator.from_mjcf(
        Path(MENAGERIE) / "universal_robots_ur5e" / "ur5e.xml", base="world", ee="wrist_3_link"
    )
    ik = SSIKSolver(arm, T_ee=site_offset_in_body(model, "attachment_site"))
    # The UR5e's joints are bounded intervals (±2π, elbow ±π), not continuous circles (#35).
    config = CBiRRTConfig(max_iterations=5000, step_size=0.2, tsr_samples=100)
    return robot, collision, ik, CBiRRT(robot, ik, collision, config)


def gripper_down_everywhere():
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T[:3, 3] = [0.0, 0.0, 0.6]
    bounds = np.array([[-0.9, 0.9], [-0.9, 0.9], [-0.3, 0.5], [-0.05, 0.05], [-0.05, 0.05], [-np.pi, np.pi]])
    return TSR(T0_w=T, Tw_e=np.eye(4), Bw=bounds)


def test_ssik_and_mujoco_forward_kinematics_agree(ur5e):
    """The frame contract: SSIK's FK (with T_ee) equals MuJoCo's attachment-site FK."""
    robot, _, ik, _ = ur5e
    rng = np.random.default_rng(0)
    for _ in range(25):
        q = rng.uniform(-np.pi, np.pi, 6)
        assert np.allclose(ik.fk(q), robot.forward_kinematics(q), atol=1e-9)


def test_ik_solutions_reach_the_mujoco_pose_and_include_windings(ur5e):
    robot, _, ik, _ = ur5e
    q = np.array([0.1, -1.2, 1.0, -0.5, 0.3, 0.2])
    T = robot.forward_kinematics(q)
    sols = ik.solve(T)
    assert len(sols) > 8  # eight geometric branches, each with in-limit windings on the ±2π joints
    lo, hi = robot.joint_limits
    for s in sols:
        assert np.all((s >= lo) & (s <= hi))
        assert np.allclose(robot.forward_kinematics(s), T, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
def test_constrained_transport_keeps_gripper_down(ur5e, seed):
    robot, collision, _, planner = ur5e
    start_tsr = create_grasp_tsr(np.array([0.55, -0.35, 0.47]))  # hard region: few collision-free branches
    goal_tsr = create_grasp_tsr(np.array([-0.30, 0.45, 0.47]))
    upright = gripper_down_everywhere()

    result = planner.plan(
        start_tsrs=[start_tsr], goal_tsrs=[goal_tsr], constraint_tsrs=[upright], seed=seed, return_details=True
    )
    assert result.success, result.failure_reason
    for q in result.path:
        assert collision.is_valid(q)
        assert upright.distance(robot.forward_kinematics(q))[0] <= planner.config.membership_tolerance
    assert start_tsr.distance(robot.forward_kinematics(result.path[0]))[0] <= planner.config.membership_tolerance
    assert goal_tsr.distance(robot.forward_kinematics(result.path[-1]))[0] <= planner.config.membership_tolerance
    # Executable as raw joint values: no waypoint outside the real limits, no jump larger than one step
    lo, hi = robot.joint_limits
    P = np.array(result.path)
    assert np.all((P >= lo) & (P <= hi))
    assert np.abs(np.diff(P, axis=0)).max() <= planner.config.step_size + 1e-9


def test_planner_keeps_a_collision_free_branch_when_others_collide(ur5e):
    """Root seeding must find the admissible IK branches even when the first ones collide."""
    robot, collision, ik, planner = ur5e
    start_tsr = create_grasp_tsr(np.array([0.55, -0.35, 0.47]))
    space = JointSpace(*robot.joint_limits)
    s = TSRConfigurationSet(start_tsr, robot, ik, space)
    rng = np.random.default_rng(0)
    mixed_draws = 0
    for _ in range(20):
        cands = s.sample(rng)
        if not cands:
            continue
        valid = [collision.is_valid(c.q) for c in cands]
        if any(valid) and not all(valid):
            mixed_draws += 1
    assert mixed_draws > 0  # the region really is adversarial: colliding and free candidates share a pose
    from pycbirrt import FiniteSet, PlanningProblem

    prob = PlanningProblem(space=planner.space, start=s, goal=FiniteSet([np.zeros(6)]), validator=collision)
    roots = planner._roots(prob, s, "Start")
    assert roots and all(collision.is_valid(r.q) for r in roots)


def test_unconstrained_baseline_violates_constraint(ur5e):
    """Sanity check that the constraint above is not vacuous for this start/goal pair."""
    robot, _, _, planner = ur5e
    result = planner.plan(
        start_tsrs=[create_grasp_tsr(np.array([0.55, -0.35, 0.47]))],
        goal_tsrs=[create_grasp_tsr(np.array([-0.30, 0.45, 0.47]))],
        seed=0,
        return_details=True,
    )
    assert result.success
    upright = gripper_down_everywhere()
    assert max(upright.distance(robot.forward_kinematics(q))[0] for q in result.path) > 1.0


def test_projection_near_a_winding_does_not_take_a_full_turn(ur5e):
    """The #36 regression: a configuration near one winding must project to that winding, not the principal one."""
    robot, _, ik, planner = ur5e
    q = np.array([0.1, -1.2, 1.0, -0.5, 0.3, 0.2 - 2 * np.pi])  # wrist 3 on its other in-limit winding
    assert planner.space.contains(q)
    box = np.array([[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], [-np.pi, np.pi]])
    T = robot.forward_kinematics(q)
    tsr = TSR(T0_w=T, Tw_e=np.eye(4), Bw=box)
    s = TSRConfigurationSet(tsr, robot, ik, planner.space)
    q_off = q + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_off[1] += 0.3  # push the pose out of the box so projection has to solve IK
    assert not s.contains(q_off)
    q_proj = s.project(q, q_off)
    assert q_proj is not None and s.contains(q_proj)
    assert planner.space.distance(q_proj, q) < 1.0  # stayed on the nearby winding, not 2π away
