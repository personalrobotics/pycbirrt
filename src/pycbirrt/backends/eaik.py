"""EAIK backend for analytical inverse kinematics."""

from typing import Protocol

import numpy as np

try:
    import eaik
except ImportError:
    raise ImportError("EAIK backend requires eaik. Install with: pip install eaik")


class CollisionChecker(Protocol):
    """Protocol for collision checking (matches pycbirrt.interfaces.CollisionChecker)."""

    def is_valid(self, q: np.ndarray) -> bool: ...


class EAIKSolver:
    """Analytical IK solver using EAIK.

    Can be used standalone (solve only) or with validation (solve_valid).
    For solve_valid, requires joint_limits and optionally a collision_checker.
    """

    def __init__(
        self,
        urdf_path: str,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        collision_checker: CollisionChecker | None = None,
    ):
        """Initialize EAIK solver from URDF.

        Args:
            urdf_path: Path to robot URDF file
            joint_limits: Optional (lower, upper) joint limit arrays for validation
            collision_checker: Optional collision checker for validation
        """
        self.robot = eaik.Robot(urdf_path)
        self.joint_limits = joint_limits
        self.collision_checker = collision_checker

    def solve(self, pose: np.ndarray) -> list[np.ndarray]:
        """Solve IK for a single end-effector pose (raw, unvalidated).

        Args:
            pose: 4x4 homogeneous transform

        Returns:
            List of all kinematic solutions (may include invalid ones)
        """
        # EAIK expects 4x4 matrix
        solutions = self.robot.ik(pose)

        # Convert to list of numpy arrays
        return [np.array(sol) for sol in solutions]

    def solve_valid(self, pose: np.ndarray) -> list[np.ndarray]:
        """Solve IK and return only valid solutions.

        Filters solutions to return only those that are:
        - Within joint limits (if joint_limits provided)
        - Collision-free (if collision_checker provided)

        Args:
            pose: 4x4 homogeneous transform

        Returns:
            List of valid joint configurations (may be empty)
        """
        solutions = self.solve(pose)

        valid = []
        for q in solutions:
            # Check joint limits if provided
            if self.joint_limits is not None:
                lower, upper = self.joint_limits
                if not (np.all(q >= lower) and np.all(q <= upper)):
                    continue

            # Check collisions if checker provided
            if self.collision_checker is not None:
                if not self.collision_checker.is_valid(q):
                    continue

            valid.append(q)

        return valid

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
