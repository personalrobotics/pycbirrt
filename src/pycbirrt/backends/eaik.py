"""EAIK backend for analytical inverse kinematics."""

from typing import Protocol

import numpy as np

try:
    from eaik.IK_DH import DhRobot
    from eaik.IK_URDF import UrdfRobot
except ImportError:
    raise ImportError("EAIK backend requires eaik. Install with: pip install eaik")


class CollisionChecker(Protocol):
    """Protocol for collision checking (matches pycbirrt.interfaces.CollisionChecker)."""

    def is_valid(self, q: np.ndarray) -> bool: ...


class EAIKSolver:
    """Analytical IK solver using EAIK.

    EAIK provides closed-form analytical IK solutions for 6-DOF manipulators,
    typically returning up to 8 solutions for a given end-effector pose.
    This is much faster and more reliable than iterative differential IK.

    Can be initialized from either:
    - URDF file (using from_urdf class method)
    - DH parameters (using from_dh class method)

    For solve_valid, requires joint_limits and optionally a collision_checker.
    """

    def __init__(
        self,
        robot,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        collision_checker: CollisionChecker | None = None,
    ):
        """Initialize EAIK solver with a robot model.

        Use from_urdf() or from_dh() class methods instead of direct instantiation.

        Args:
            robot: EAIK robot model (DhRobot or UrdfRobot)
            joint_limits: Optional (lower, upper) joint limit arrays for validation
            collision_checker: Optional collision checker for validation
        """
        self.robot = robot
        self.joint_limits = joint_limits
        self.collision_checker = collision_checker

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        collision_checker: CollisionChecker | None = None,
    ) -> "EAIKSolver":
        """Create EAIK solver from URDF file.

        Args:
            urdf_path: Path to robot URDF file
            joint_limits: Optional (lower, upper) joint limit arrays for validation
            collision_checker: Optional collision checker for validation

        Returns:
            EAIKSolver instance
        """
        robot = UrdfRobot(urdf_path)
        return cls(robot, joint_limits, collision_checker)

    @classmethod
    def from_dh(
        cls,
        dh_a: np.ndarray,
        dh_d: np.ndarray,
        dh_alpha: np.ndarray,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        collision_checker: CollisionChecker | None = None,
    ) -> "EAIKSolver":
        """Create EAIK solver from DH parameters.

        Args:
            dh_a: Link lengths (a_i) in meters
            dh_d: Link offsets (d_i) in meters
            dh_alpha: Link twist angles (alpha_i) in radians
            joint_limits: Optional (lower, upper) joint limit arrays for validation
            collision_checker: Optional collision checker for validation

        Returns:
            EAIKSolver instance
        """
        robot = DhRobot(dh_alpha, dh_a, dh_d)
        return cls(robot, joint_limits, collision_checker)

    @classmethod
    def for_ur5e(
        cls,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        collision_checker: CollisionChecker | None = None,
    ) -> "EAIKSolver":
        """Create EAIK solver pre-configured for UR5e.

        Uses official UR5e DH parameters from Universal Robots.

        Args:
            joint_limits: Optional (lower, upper) joint limit arrays for validation
            collision_checker: Optional collision checker for validation

        Returns:
            EAIKSolver instance configured for UR5e
        """
        # UR5e DH parameters from Universal Robots official documentation
        # Joint  a [m]    d [m]     alpha [rad]
        # 1      0        0.1625    π/2
        # 2     -0.425    0         0
        # 3     -0.3922   0         0
        # 4      0        0.1333    π/2
        # 5      0        0.0997   -π/2
        # 6      0        0.0996    0
        dh_a = np.array([0, -0.425, -0.3922, 0, 0, 0])
        dh_d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
        dh_alpha = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])

        return cls.from_dh(dh_a, dh_d, dh_alpha, joint_limits, collision_checker)

    def has_known_decomposition(self) -> bool:
        """Check if EAIK has a known analytical solution for this robot."""
        return self.robot.hasKnownDecomposition()

    def get_kinematic_family(self) -> str:
        """Get the kinematic family of this robot."""
        return self.robot.getKinematicFamily()

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute forward kinematics.

        Args:
            q: Joint configuration

        Returns:
            4x4 homogeneous transform of end-effector
        """
        return self.robot.fwdKin(q)

    def solve(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Solve IK for a single end-effector pose (raw, unvalidated).

        Note: q_init is accepted for interface compatibility but ignored,
        as EAIK is an analytical solver that finds all solutions regardless
        of initial configuration.

        Args:
            pose: 4x4 homogeneous transform
            q_init: Ignored (for interface compatibility with iterative solvers)

        Returns:
            List of all kinematic solutions (may include invalid ones)
        """
        # EAIK returns IKSolution object with Q matrix of shape (N, dof)
        ik_result = self.robot.IK(pose)
        num_solutions = ik_result.num_solutions()

        if num_solutions == 0:
            return []

        return [ik_result.Q[i] for i in range(num_solutions)]

    def solve_valid(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Solve IK and return only valid solutions.

        Filters solutions to return only those that are:
        - Within joint limits (if joint_limits provided)
        - Collision-free (if collision_checker provided)

        Note: q_init is accepted for interface compatibility but ignored,
        as EAIK is an analytical solver that finds all solutions regardless
        of initial configuration.

        Args:
            pose: 4x4 homogeneous transform
            q_init: Ignored (for interface compatibility with iterative solvers)

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
        # Convert to list of 4x4 matrices for EAIK
        pose_list = [poses[i] for i in range(poses.shape[0])]
        results = self.robot.IK_batched(pose_list)

        all_solutions = []
        for ik_result in results:
            num_sol = ik_result.num_solutions()
            solutions = [ik_result.Q[i] for i in range(num_sol)]
            all_solutions.append(solutions)

        return all_solutions
