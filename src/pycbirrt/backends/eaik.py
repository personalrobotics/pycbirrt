"""EAIK backend for analytical inverse kinematics."""

import numpy as np

try:
    import eaik
except ImportError:
    raise ImportError("EAIK backend requires eaik. Install with: pip install eaik")


class EAIKSolver:
    """Analytical IK solver using EAIK."""

    def __init__(self, urdf_path: str):
        """Initialize EAIK solver from URDF.

        Args:
            urdf_path: Path to robot URDF file
        """
        self.robot = eaik.Robot(urdf_path)

    def solve(self, pose: np.ndarray) -> list[np.ndarray]:
        """Solve IK for a single end-effector pose.

        Args:
            pose: 4x4 homogeneous transform

        Returns:
            List of all valid joint configurations (may be empty)
        """
        # EAIK expects 4x4 matrix
        solutions = self.robot.ik(pose)

        # Convert to list of numpy arrays
        return [np.array(sol) for sol in solutions]

    def solve_batch(self, poses: np.ndarray) -> list[list[np.ndarray]]:
        """Solve IK for multiple poses using batched computation.

        Args:
            poses: Array of shape (N, 4, 4)

        Returns:
            List of N lists of solutions
        """
        # EAIK has batched IK support
        results = self.robot.ik_batch(poses)

        return [[np.array(sol) for sol in sols] for sols in results]
