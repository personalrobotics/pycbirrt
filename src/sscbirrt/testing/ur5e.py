# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The UR5e test scene: a UR5e with a Robotiq 2F-85, a table, and a cylinder.

The reference artifact's UR5e cases (tools/reference_artifact.py) and the UR5e tests plan in this exact scene, so its
geometry is frozen: it moved here unchanged from the former examples/ur5e_mujoco.py (#166). The demos use their own
scene (``sscbirrt.demo.scene``). Requires mujoco.
"""

from pathlib import Path

import mujoco
import numpy as np
from tsr import TSR

JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
HOME = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


def create_grasp_tsr(target_pos: np.ndarray) -> TSR:
    """Create a TSR for top-down grasp of a cylinder.

    The gripper approaches from above with z-axis pointing DOWN.
    Full yaw freedom allows grasping from any angle around the vertical axis.

    Geometry: attachment_site is 0.25m above the cylinder center,
    accounting for gripper length (~0.145m) and clearance.
    """
    standoff = 0.25  # Height above cylinder center for attachment_site

    T0_w = np.eye(4)
    T0_w[:3, 3] = target_pos + np.array([0, 0, standoff])
    # Gripper z points DOWN (180 deg rotation around x)
    T0_w[:3, :3] = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )

    return TSR(
        T0_w=T0_w,
        Tw_e=np.eye(4),
        Bw=np.array(
            [
                [-0.02, 0.02],  # x tolerance
                [-0.02, 0.02],  # y tolerance
                [0, 0.05],  # z: can be 0-5cm higher
                [-0.01, 0.01],  # roll: small tolerance
                [-0.01, 0.01],  # pitch: small tolerance
                [-np.pi, np.pi],  # yaw: full rotation
            ]
        ),
    )


def create_scene(menagerie_path: Path | None = None, free_cylinder: bool = False) -> "mujoco.MjModel":
    """Create a MuJoCo model with UR5e, Robotiq gripper, table, and cylinder.

    Uses MuJoCo's attach mechanism to connect the gripper to the robot arm. ``menagerie_path`` defaults to
    ``MUJOCO_MENAGERIE_PATH`` if set, else the sscbirrt-assets wheel.
    """
    if menagerie_path is None:
        from sscbirrt.demo.scene import menagerie_path as default_path

        menagerie_path = default_path()
    ur5e_path = menagerie_path / "universal_robots_ur5e" / "ur5e.xml"
    robotiq_path = menagerie_path / "robotiq_2f85" / "2f85.xml"

    if not ur5e_path.exists():
        raise FileNotFoundError(f"UR5e model not found at {ur5e_path}")
    if not robotiq_path.exists():
        raise FileNotFoundError(f"Robotiq model not found at {robotiq_path}")

    # Load both models
    ur5e_spec = mujoco.MjSpec.from_file(str(ur5e_path))
    gripper_spec = mujoco.MjSpec.from_file(str(robotiq_path))

    # High-res offscreen rendering
    ur5e_spec.visual.global_.offwidth = 1920
    ur5e_spec.visual.global_.offheight = 1080

    # Attach gripper to UR5e wrist
    wrist_body = ur5e_spec.body("wrist_3_link")

    attachment_site = None
    for site in wrist_body.sites:
        if site.name == "attachment_site":
            attachment_site = site
            break
    if attachment_site is None:
        raise RuntimeError("Could not find attachment_site on wrist_3_link")

    gripper_base = gripper_spec.body("base_mount")
    frame = wrist_body.add_frame()
    frame.name = "gripper_attachment_frame"
    frame.pos = attachment_site.pos
    frame.quat = attachment_site.quat
    # "gripper_" prefix avoids name collisions (both models have "black" material)
    frame.attach_body(gripper_base, "gripper_", "")

    # Scene elements
    world = ur5e_spec.worldbody

    # Ground plane
    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.pos = [0, 0, -0.01]
    floor.size = [0, 0, 0.05]
    floor.rgba = [0.95, 0.95, 0.95, 1]

    # Robot base marker (visual only)
    base_marker = world.add_geom()
    base_marker.name = "base_marker"
    base_marker.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    base_marker.pos = [0, 0, 0.01]
    base_marker.size = [0.1, 0.01, 0]
    base_marker.rgba = [0.3, 0.5, 0.7, 0.8]
    base_marker.contype = 0
    base_marker.conaffinity = 0

    # Table
    table_body = world.add_body()
    table_body.name = "table"
    table_body.pos = [0.5, 0, 0.4]  # Table top at z=0.42

    table_top = table_body.add_geom()
    table_top.name = "table_top"
    table_top.type = mujoco.mjtGeom.mjGEOM_BOX
    table_top.size = [0.25, 0.4, 0.02]
    table_top.rgba = [0.4, 0.3, 0.2, 1]

    # Table legs (visual only)
    for i, (x, y) in enumerate([(0.22, 0.37), (0.22, -0.37), (-0.22, 0.37), (-0.22, -0.37)]):
        leg = table_body.add_geom()
        leg.name = f"table_leg_{i}"
        leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        leg.pos = [x, y, -0.20]
        leg.size = [0.025, 0.18, 0]
        leg.rgba = [0.3, 0.25, 0.2, 1]
        leg.contype = 0
        leg.conaffinity = 0

    # Target cylinder on table
    cylinder_body = world.add_body()
    cylinder_body.name = "cylinder"
    cylinder_body.pos = [0.45, 0.15, 0.47]  # On table surface + half height
    if free_cylinder:
        cylinder_body.add_freejoint()  # a movable object, so it can be grasped and carried (native scene attachments)

    cylinder_geom = cylinder_body.add_geom()
    cylinder_geom.name = "cylinder_geom"
    cylinder_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    cylinder_geom.size = [0.03, 0.05, 0]  # radius=3cm, half-height=5cm
    cylinder_geom.rgba = [1, 0.2, 0.2, 1]

    return ur5e_spec.compile()
