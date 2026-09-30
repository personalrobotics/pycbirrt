# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The analytical IK the demos plan with: SSIK built from the same UR5e MJCF as the scene."""

from __future__ import annotations

import mujoco
import numpy as np

from sscbirrt.backends.ssik import SSIKSolver
from sscbirrt.demo.scene import EE_SITE, ur5e_xml


def _pose(xpos: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = xmat.reshape(3, 3)
    T[:3, 3] = xpos
    return T


def ee_offset(model: mujoco.MjModel, body: str = "wrist_3_link", site: str = EE_SITE) -> np.ndarray:
    """The fixed transform from the SSIK chain's last body to the grasp site, which lives on the gripper."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    T_body = _pose(data.body(body).xpos, data.body(body).xmat)
    T_site = _pose(data.site(site).xpos, data.site(site).xmat)
    return np.linalg.inv(T_body) @ T_site


def build_ik(model: mujoco.MjModel) -> SSIKSolver:
    """SSIK with the world as base and the grasp site as the end effector.

    Built from an ``ssik.Manipulator`` (not a prebuilt module) so it has a native form: the native
    backend then samples and projects the grasp regions in C++.
    """
    import ssik

    arm = ssik.Manipulator.from_mjcf(str(ur5e_xml()), base="world", ee="wrist_3_link")
    return SSIKSolver(arm, T_ee=ee_offset(model))
