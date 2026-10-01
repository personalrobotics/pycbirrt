# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Grasp regions for the demo cans, from the tsr package's Robotiq 2F-85 cylinder primitive."""

from __future__ import annotations

import numpy as np
from tsr import TSR, Robotiq2F85

from sscbirrt.demo.scene import CAN_HALF_HEIGHT, CAN_RADIUS


def side_grasps(center: np.ndarray) -> list[TSR]:
    """Every side grasp of the can centered at ``center``: three depths by two finger rolls, any approach angle.

    The primitive's reference frame is the cylinder's bottom face; the TSRs are in the 2F-85's grasp frame
    (the scene's ``EE_SITE``).
    """
    T_bottom = np.eye(4)
    T_bottom[:3, 3] = np.asarray(center) - [0.0, 0.0, CAN_HALF_HEIGHT]
    templates = Robotiq2F85().grasp_cylinder_side(cylinder_radius=CAN_RADIUS, cylinder_height=2 * CAN_HALF_HEIGHT)
    return [t.instantiate(T_bottom) for t in templates]
