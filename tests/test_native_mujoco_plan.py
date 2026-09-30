# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The one-call native path from a live MuJoCo world (#88, #140)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from pycbirrt.backends import native_mujoco as nm  # noqa: E402
from pycbirrt.backends.native import NativeUnsupported  # noqa: E402

if not nm.available():
    pytest.skip(nm.unavailable_reason(), allow_module_level=True)

from pycbirrt import CBiRRTConfig  # noqa: E402

XML = """
<mujoco><compiler angle="radian"/><worldbody>
  <geom name="floor" type="plane" size="3 3 0.1"/>
  <geom name="wall" type="box" pos="0.47 0 0.3" size="0.05 0.5 0.3"/>
  <body name="arm/base" pos="0 0 0.1"><joint name="j0" type="hinge" axis="0 0 1" limited="true" range="-3 3"/>
    <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
    <body name="arm/link1" pos="0 0 0.1"><joint name="j1" type="hinge" axis="0 1 0" limited="true" range="-2 2"/>
      <geom type="capsule" size="0.03" fromto="0 0 0 0.4 0 0"/>
      <body name="arm/gripper/base" pos="0.4 0 0">
        <geom type="box" size="0.02 0.03 0.02"/><site name="tip" pos="0.02 0 0"/>
      </body>
    </body>
  </body>
</worldbody></mujoco>"""


def test_plan_native_finite_problem_on_a_small_world():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    r = nm.plan_native(
        model,
        data,
        ["j0", "j1"],
        start=np.array([0.0, -1.2]),
        goal=np.array([2.5, -1.2]),
        ee_site="tip",
        config=CBiRRTConfig(step_size=0.1, edge_resolution=0.02, timeout=30.0),
        seed=0,
    )
    assert r.success and r.backend == "native"
    assert r.provenance["snapshot_sha256"] and r.provenance["scene_mjb_sha256"] and r.provenance["mujoco"] == "3.14.0"
    assert r.stats["state_checks"] > 0
    # A Python validator in the loop is refused with a reason naming it.
    from pycbirrt import CBiRRT
    from pycbirrt.backends.mujoco import MuJoCoRobotModel

    class PythonChecker:
        def is_valid(self, q):
            return True

    planner = CBiRRT(
        MuJoCoRobotModel(model, data, "tip", ["j0", "j1"]),
        nm._NoIK(),
        PythonChecker(),
        CBiRRTConfig(),
        backend="native",
    )
    with pytest.raises(NativeUnsupported, match="validator: PythonChecker is a Python object"):
        planner.plan(start=np.array([0.0, -1.2]), goal=np.array([2.5, -1.2]), seed=0)


@pytest.mark.skipif("MUJOCO_MENAGERIE_PATH" not in os.environ, reason="needs the MuJoCo Menagerie")
class TestUR5eReleaseCases:
    @pytest.fixture(scope="class")
    def world(self):
        ssik = pytest.importorskip("ssik")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
        from ur5e_mujoco import create_grasp_tsr, create_scene

        from pycbirrt.backends.mujoco import site_offset_in_body
        from pycbirrt.backends.ssik import SSIKSolver

        menagerie = Path(os.environ["MUJOCO_MENAGERIE_PATH"])
        model = create_scene(menagerie, free_cylinder=True)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        arm = ssik.Manipulator.from_mjcf(
            str(menagerie / "universal_robots_ur5e" / "ur5e.xml"), base="world", ee="wrist_3_link"
        )
        ik = SSIKSolver(arm, T_ee=site_offset_in_body(model, "attachment_site"))
        joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        return model, data, ik, joints, create_grasp_tsr

    def test_tsr_goal_among_obstacles_runs_natively(self, world):
        model, data, ik, joints, grasp = world
        home = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        r = nm.plan_native(
            model,
            data,
            joints,
            ik=ik,
            start=home,
            goal_tsrs=[grasp(model.body("cylinder").pos.copy())],
            config=CBiRRTConfig(step_size=0.2, edge_resolution=0.05, timeout=60.0, num_tree_roots=20),
            seed=20,
        )
        assert r.success and r.backend == "native"
        assert r.provenance["ssik_solver_name"] == "ikgeo.three_parallel" and r.provenance["snapshot_sha256"]
        scene = nm.NativeScene.from_model(model, joints)
        checker = nm.NativeCollisionChecker(scene, nm.Snapshot.capture(scene, data))
        assert all(checker.is_valid(q) for q in r.path)

    def test_held_object_runs_natively_with_the_grasp_allowed(self, world):
        model, data, ik, joints, grasp = world
        home = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        for i, a in enumerate(model.jnt_qposadr[[model.joint(j).id for j in joints]]):
            data.qpos[a] = home[i]
        mujoco.mj_forward(model, data)
        T = np.eye(4)
        T[:3, 3] = [0.0, 0.0, 0.16]
        attachments = {"cylinder": ("gripper_base_mount", T)}
        r = nm.plan_native(
            model,
            data,
            joints,
            ik=ik,
            start=home,
            goal_tsrs=[grasp(np.array([0.5, -0.25, 0.47]))],
            attachments=attachments,
            config=CBiRRTConfig(step_size=0.2, edge_resolution=0.05, timeout=60.0, num_tree_roots=20),
            seed=21,
        )
        assert r.success and r.backend == "native"
        scene = nm.NativeScene.from_model(model, joints)
        checker = nm.NativeCollisionChecker(scene, nm.Snapshot.capture(scene, data, attachments=attachments))
        assert all(checker.is_valid(q) for q in r.path)
        # the grasp itself is a contact, and it is allowed: without the attachment the held pose is not special
        assert checker.invalid_contacts(home) == []
