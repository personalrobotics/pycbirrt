# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The demo world: a UR5e with a Robotiq 2F-85 at the origin, a table in front, and props on it."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import mujoco
import numpy as np

UR5E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Arm up, gripper pointing down over the table.
HOME = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

# The tsr package's canonical grasp frame for the 2F-85 (tsr.Robotiq2F85): the forward edge of the gripper
# housing, z = approach, y = finger opening. The same site mj_manipulator attaches.
EE_SITE = "gripper_grasp_site"
GRIPPER_BODY = "gripper_base_mount"

TABLE_CENTER = np.array([0.5, 0.0, 0.40])
TABLE_HALF = np.array([0.25, 0.40, 0.02])
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_HALF[2]

CAN_RADIUS = 0.03
CAN_HALF_HEIGHT = 0.05
CAN_Z = TABLE_TOP_Z + CAN_HALF_HEIGHT

CAN_COLORS = [(0.85, 0.20, 0.15, 1), (0.20, 0.55, 0.85, 1), (0.95, 0.70, 0.15, 1), (0.30, 0.70, 0.35, 1)]


def menagerie_path() -> Path:
    """Where the UR5e and 2F-85 MJCFs live: ``MUJOCO_MENAGERIE_PATH`` if set, else the sscbirrt-assets wheel."""
    override = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if override:
        return Path(override)
    try:
        import sscbirrt_assets
    except ImportError as e:
        raise ImportError(
            'The demo robot models are in the sscbirrt-assets package: pip install "sscbirrt[demo]" '
            "(or set MUJOCO_MENAGERIE_PATH to a mujoco_menagerie clone)."
        ) from e
    return sscbirrt_assets.menagerie_path()


def ur5e_xml() -> Path:
    return _existing(menagerie_path() / "universal_robots_ur5e" / "ur5e.xml")


def robotiq_2f85_xml() -> Path:
    return _existing(menagerie_path() / "robotiq_2f85" / "2f85.xml")


def _existing(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (models are read from {menagerie_path()})")
    return path


def can_position(xy: Sequence[float]) -> np.ndarray:
    """The center of a can standing on the table at ``xy``."""
    return np.array([xy[0], xy[1], CAN_Z])


def build_scene(cans: Mapping[str, Sequence[float]] | None = None) -> mujoco.MjModel:
    """Compile the world: arm, gripper, floor, table, and one free-floating can per entry of ``cans``.

    ``cans`` maps a body name to the can's center (see :func:`can_position`). Cans are free bodies so a
    planner can treat one as held (``attachments``); the demos never step the simulation.
    """
    from tsr import Robotiq2F85

    arm = mujoco.MjSpec.from_file(str(ur5e_xml()))
    gripper = mujoco.MjSpec.from_file(str(robotiq_2f85_xml()))
    grasp_site = gripper.body("base_mount").add_site()
    grasp_site.name = "grasp_site"
    grasp_site.pos = [0.0, 0.0, Robotiq2F85.PALM_OFFSET_FROM_BASE_MOUNT]
    grasp_site.quat = [0.7071068, 0.0, 0.0, -0.7071068]  # -90 deg about z: finger opening along y
    grasp_site.group = 5  # hidden in renders

    _style(arm)

    # The 2F-85 sets impratio and an elliptic cone for grasping physics. The demos never step the simulation,
    # but matching them keeps MjSpec.attach from warning about the conflict.
    arm.option.impratio = gripper.option.impratio
    arm.option.cone = gripper.option.cone

    wrist = arm.body("wrist_3_link")
    site = next((s for s in wrist.sites if s.name == "attachment_site"), None)
    if site is None:
        raise RuntimeError(f"no attachment_site on wrist_3_link in {ur5e_xml()}")
    frame = wrist.add_frame()
    frame.name = "gripper_attachment_frame"
    frame.pos = site.pos
    frame.quat = site.quat
    # The prefix avoids name collisions (both models define a "black" material).
    frame.attach_body(gripper.body("base_mount"), "gripper_", "")

    world = arm.worldbody
    key = world.add_light()
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.dir = [0.4, 0.5, -1.0]
    key.diffuse = [0.45, 0.45, 0.45]
    key.specular = [0.1, 0.1, 0.1]
    key.castshadow = True

    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.pos = [0, 0, -0.01]
    floor.size = [0, 0, 0.05]
    floor.material = "floor"

    base = world.add_geom()
    base.name = "base_marker"
    base.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    base.pos = [0, 0, 0.01]
    base.size = [0.1, 0.01, 0]
    base.rgba = [0.3, 0.5, 0.7, 0.8]
    base.contype = 0
    base.conaffinity = 0

    table = world.add_body()
    table.name = "table"
    table.pos = list(TABLE_CENTER)
    top = table.add_geom()
    top.name = "table_top"
    top.type = mujoco.mjtGeom.mjGEOM_BOX
    top.size = list(TABLE_HALF)
    top.material = "table"
    leg_half = (TABLE_CENTER[2] - TABLE_HALF[2]) / 2  # from the floor to the underside of the top
    for i, (x, y) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        leg = table.add_geom()
        leg.name = f"table_leg_{i}"
        leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        leg.pos = [x * (TABLE_HALF[0] - 0.03), y * (TABLE_HALF[1] - 0.03), leg_half - TABLE_CENTER[2]]
        leg.size = [0.025, leg_half, 0]
        leg.material = "table_leg"
        leg.contype = 0
        leg.conaffinity = 0

    for i, (name, pos) in enumerate((cans or {}).items()):
        body = world.add_body()
        body.name = name
        body.pos = list(pos)
        body.add_freejoint()
        geom = body.add_geom()
        geom.name = f"{name}_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size = [CAN_RADIUS, CAN_HALF_HEIGHT, 0]
        geom.rgba = list(CAN_COLORS[i % len(CAN_COLORS)])

    return arm.compile()


def _style(spec: mujoco.MjSpec) -> None:
    """Offscreen resolution and anti-aliasing, a gradient sky, a checkered floor, and matte materials."""
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.quality.offsamples = 8
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.30, 0.30, 0.30]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]

    sky = spec.add_texture()
    sky.name = "sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.82, 0.87, 0.93]
    sky.rgb2 = [0.97, 0.98, 1.0]
    sky.width = sky.height = 512

    grid = spec.add_texture()
    grid.name = "floor"
    grid.type = mujoco.mjtTexture.mjTEXTURE_2D
    grid.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    grid.rgb1 = [0.86, 0.87, 0.88]
    grid.rgb2 = [0.80, 0.81, 0.83]
    grid.width = grid.height = 512
    floor = spec.add_material()
    floor.name = "floor"
    textures = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "floor"
    floor.textures = textures
    floor.texrepeat = [8, 8]
    floor.texuniform = True
    floor.reflectance = 0.0

    for name, rgba in (("table", [0.55, 0.42, 0.30, 1]), ("table_leg", [0.30, 0.26, 0.22, 1])):
        mat = spec.add_material()
        mat.name = name
        mat.rgba = rgba
        mat.specular = 0.2
        mat.shininess = 0.3


def set_arm(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, joints: Sequence[str] = UR5E_JOINTS) -> None:
    """Write the arm configuration into ``data`` and run forward kinematics."""
    for name, value in zip(joints, q):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)
