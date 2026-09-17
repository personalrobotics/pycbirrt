# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from gafro import Motor


class RobotModel(Protocol):
    """Protocol for robot kinematics and joint limits."""

    @property
    def dof(self) -> int:
        """Number of degrees of freedom."""
        ...

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint limits as (lower, upper) bounds arrays."""
        ...

    def forward_kinematics(self, q: np.ndarray) -> "Motor":
        """Compute end-effector pose from joint configuration.

        Args:
            q: Joint configuration array of shape (dof,)

        Returns:
            End-effector pose in the world frame as a ``gafro.Motor``.
            (The planner also tolerates a 4x4 homogeneous transform, which it
            normalizes via ``as_motor``, but Motor is the contract.)
        """
        ...

    def normalize_pose(self, x):
        """Coerce a pose (FK output or TSR pose) to this model's pose token.

        Single-arm models return a ``gafro.Motor``; a bimanual model returns
        a ``tsr.bimanual.BimanualPose``. The planner treats the result as opaque.
        """
        ...
