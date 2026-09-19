# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The UR5e examples' MuJoCo differential-IK fallback works when forced (#65).

Needs mujoco and MUJOCO_MENAGERIE_PATH, not ssik: the point is to exercise the
non-SSIK path even in an environment where SSIK is installed.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
MENAGERIE = os.environ.get("MUJOCO_MENAGERIE_PATH")
if not MENAGERIE or not (Path(MENAGERIE) / "universal_robots_ur5e").exists():
    pytest.skip("MUJOCO_MENAGERIE_PATH not set to a mujoco_menagerie clone", allow_module_level=True)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))

import tsr_union_demo  # noqa: E402
import ur5e_mujoco  # noqa: E402

from pycbirrt import CBiRRT, CBiRRTConfig  # noqa: E402
from pycbirrt.backends.mujoco import MuJoCoCollisionChecker, MuJoCoIKSolver, MuJoCoRobotModel  # noqa: E402

JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
HOME = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])


class TestUr5eMujocoExample:
    def test_forced_mujoco_fallback_solves_a_nontrivial_pose(self):
        model = ur5e_mujoco.create_scene(Path(MENAGERIE))
        data = mujoco.MjData(model)
        robot = MuJoCoRobotModel(model, data, "attachment_site", JOINTS)
        collision = MuJoCoCollisionChecker(model, data, JOINTS)
        ik, name = ur5e_mujoco.build_ik_solver(model, data, JOINTS, collision, Path(MENAGERIE), backend="mujoco")
        assert name == "mujoco" and isinstance(ik, MuJoCoIKSolver)
        assert ik.collision_checker is collision
        target = robot.forward_kinematics(HOME + np.array([0.3, 0.2, -0.2, 0.1, 0.1, 0.4]))
        sols = ik.solve(target, q_init=HOME)  # requires iterative updates from the seed
        assert sols
        assert np.linalg.norm(robot.forward_kinematics(sols[0])[:3, 3] - target[:3, 3]) < 5e-3

    def test_auto_prefers_ssik_when_installed(self):
        pytest.importorskip("ssik")
        from pycbirrt.backends.ssik import SSIKSolver

        model = ur5e_mujoco.create_scene(Path(MENAGERIE))
        data = mujoco.MjData(model)
        collision = MuJoCoCollisionChecker(model, data, JOINTS)
        ik, name = ur5e_mujoco.build_ik_solver(model, data, JOINTS, collision, Path(MENAGERIE), backend="auto")
        assert name == "ssik" and isinstance(ik, SSIKSolver)

    def test_bad_backend_rejected(self):
        with pytest.raises(ValueError):
            ur5e_mujoco.build_ik_solver(None, None, JOINTS, None, Path(MENAGERIE), backend="eaik")

    def test_script_runs_through_the_mujoco_fallback(self, tmp_path):
        env = dict(os.environ, MPLBACKEND="Agg", MUJOCO_MENAGERIE_PATH=MENAGERIE)
        proc = subprocess.run(
            [sys.executable, str(EXAMPLES / "ur5e_mujoco.py"), "--no-viz", "--ik", "mujoco"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "Using MuJoCo (differential) IK solver" in proc.stdout
        assert "Found path" in proc.stdout


class TestTsrUnionDemo:
    def test_one_planning_query_through_the_mujoco_fallback(self):
        target = np.array([0.45, 0.0, 0.38])
        model = tsr_union_demo.create_scene(Path(MENAGERIE), target)
        data = mujoco.MjData(model)
        robot = MuJoCoRobotModel(model, data, "attachment_site", JOINTS)
        collision = MuJoCoCollisionChecker(model, data, JOINTS)
        ik, name = tsr_union_demo.build_ik_solver(model, data, JOINTS, collision, Path(MENAGERIE), backend="mujoco")
        assert name == "mujoco" and isinstance(ik, MuJoCoIKSolver)
        planner = CBiRRT(robot, ik, collision, CBiRRTConfig(timeout=60.0, goal_bias=0.15, tsr_samples=100))
        top, side = tsr_union_demo.create_grasp_tsrs(target)
        result = planner.plan(start=HOME, goal_tsrs=[top, side], seed=0, return_details=True)
        assert result.success, result.failure_reason
        assert all(collision.is_valid(q) for q in result.path)
