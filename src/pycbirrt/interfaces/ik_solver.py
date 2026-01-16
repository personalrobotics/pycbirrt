from typing import Protocol
import numpy as np


class IKSolver(Protocol):
    """Protocol for inverse kinematics solvers."""

    def solve(self, pose: np.ndarray) -> list[np.ndarray]:
        """Solve IK for a single end-effector pose.

        Args:
            pose: 4x4 homogeneous transform of desired end-effector pose

        Returns:
            List of valid joint configurations (may be empty if no solution)
        """
        ...

    def solve_batch(self, poses: np.ndarray) -> list[list[np.ndarray]]:
        """Solve IK for multiple end-effector poses.

        Args:
            poses: Array of shape (N, 4, 4) of desired end-effector poses

        Returns:
            List of N lists, each containing valid joint configurations
        """
        ...
