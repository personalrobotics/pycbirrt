# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Integration check on a real arm: constrained planning with the UR5e in MuJoCo.

Skipped unless mujoco and eaik are installed and MUJOCO_MENAGERIE_PATH points
at a clone of google-deepmind/mujoco_menagerie. This is the check that caught
the root-sampling regression in #33: the start region below has four IK
branches per pose of which only one is collision-free, and never the first.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("eaik")
MENAGERIE = os.environ.get("MUJOCO_MENAGERIE_PATH")
if not MENAGERIE or not (Path(MENAGERIE) / "universal_robots_ur5e").exists():
    pytest.skip("MUJOCO_MENAGERIE_PATH not set to a mujoco_menagerie clone", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from tsr import TSR  # noqa: E402
from ur5e_mujoco import create_grasp_tsr, create_scene  # noqa: E402

from pycbirrt import CBiRRT, CBiRRTConfig  # noqa: E402
from pycbirrt.backends.eaik import EAIKSolver  # noqa: E402
from pycbirrt.backends.mujoco import MuJoCoCollisionChecker, MuJoCoRobotModel  # noqa: E402

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
    ik = EAIKSolver.for_ur5e(robot.joint_limits, collision)
    # The UR5e's joints are bounded intervals (±2π, elbow ±π), not continuous circles, so they are
    # modeled with limits rather than as angular joints; see #35.
    config = CBiRRTConfig(max_iterations=5000, step_size=0.2, tsr_samples=100)
    return robot, collision, CBiRRT(robot, ik, collision, config)


def gripper_down_everywhere():
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T[:3, 3] = [0.0, 0.0, 0.6]
    bounds = np.array([[-0.9, 0.9], [-0.9, 0.9], [-0.3, 0.5], [-0.05, 0.05], [-0.05, 0.05], [-np.pi, np.pi]])
    return TSR(T0_w=T, Tw_e=np.eye(4), Bw=bounds)


@pytest.mark.parametrize("seed", [0, 1])
def test_constrained_transport_keeps_gripper_down(ur5e, seed):
    robot, collision, planner = ur5e
    start_tsr = create_grasp_tsr(np.array([0.55, -0.35, 0.47]))  # hard region: 1 valid IK branch of 4
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


def test_unconstrained_baseline_violates_constraint(ur5e):
    """Sanity check that the constraint above is not vacuous for this start/goal pair."""
    robot, _, planner = ur5e
    result = planner.plan(
        start_tsrs=[create_grasp_tsr(np.array([0.55, -0.35, 0.47]))],
        goal_tsrs=[create_grasp_tsr(np.array([-0.30, 0.45, 0.47]))],
        seed=0,
        return_details=True,
    )
    assert result.success
    upright = gripper_down_everywhere()
    assert max(upright.distance(robot.forward_kinematics(q))[0] for q in result.path) > 1.0
