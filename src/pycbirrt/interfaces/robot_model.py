from typing import Protocol
import numpy as np


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

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector pose from joint configuration.

        Args:
            q: Joint configuration array of shape (dof,)

        Returns:
            4x4 homogeneous transform of end-effector in world frame
        """
        ...
