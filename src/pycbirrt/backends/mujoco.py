# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""MuJoCo backend for robot model, collision checking, and IK."""

import numpy as np

from pycbirrt.interfaces.collision_checker import CollisionChecker

try:
    import mujoco
except ImportError:
    raise ImportError("MuJoCo backend requires mujoco. Install with: pip install mujoco")


class MuJoCoRobotModel:
    """Robot model using MuJoCo for forward kinematics."""

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        ee_site: str,
        joint_names: list[str] | None = None,
    ):
        """Initialize MuJoCo robot model.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            ee_site: Name of end-effector site
            joint_names: List of joint names to control (if None, uses all joints)
        """
        self.model = model
        self.data = data
        self.ee_site = ee_site

        # Get site ID
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{ee_site}' not found in model")

        # Get joint indices
        if joint_names is not None:
            self.joint_ids = []
            for name in joint_names:
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jnt_id == -1:
                    raise ValueError(f"Joint '{name}' not found in model")
                self.joint_ids.append(jnt_id)
            self.joint_ids = np.array(self.joint_ids)
        else:
            # Use all joints
            self.joint_ids = np.arange(model.njnt)

        self._dof = len(self.joint_ids)

        # Cache joint limits
        self._lower = self.model.jnt_range[self.joint_ids, 0].copy()
        self._upper = self.model.jnt_range[self.joint_ids, 1].copy()

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector pose from joint configuration.

        Args:
            q: Joint configuration

        Returns:
            4x4 homogeneous transform
        """
        # Set joint positions
        for i, jnt_id in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[qpos_adr] = q[i]

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Get site pose
        pos = self.data.site_xpos[self.ee_site_id].copy()
        rot_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

        # Build 4x4 transform
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = pos

        return transform


class MuJoCoCollisionChecker:
    """Collision checker using MuJoCo."""

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        joint_names: list[str] | None = None,
    ):
        """Initialize MuJoCo collision checker.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            joint_names: List of joint names (must match robot model)
        """
        self.model = model
        self.data = data

        # Get joint indices
        if joint_names is not None:
            self.joint_ids = []
            for name in joint_names:
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jnt_id == -1:
                    raise ValueError(f"Joint '{name}' not found in model")
                self.joint_ids.append(jnt_id)
            self.joint_ids = np.array(self.joint_ids)
        else:
            self.joint_ids = np.arange(model.njnt)

    def is_valid(self, q: np.ndarray) -> bool:
        """Check if configuration is collision-free.

        Args:
            q: Joint configuration

        Returns:
            True if collision-free
        """
        # Set joint positions
        for i, jnt_id in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[qpos_adr] = q[i]

        # Run collision detection
        mujoco.mj_forward(self.model, self.data)

        # Check for contacts
        return self.data.ncon == 0

    def is_valid_batch(self, qs: np.ndarray) -> np.ndarray:
        """Check multiple configurations for collisions.

        Args:
            qs: Array of shape (N, dof)

        Returns:
            Boolean array of shape (N,)
        """
        return np.array([self.is_valid(q) for q in qs])


class MuJoCoIKSolver:
    """Differential IK solver using MuJoCo's Jacobian.

    Uses damped least squares (Levenberg-Marquardt) for numerical stability.
    Unlike analytical solvers (e.g., EAIK), this returns at most one solution
    per call, found iteratively from the current configuration.

    To find multiple solutions, call solve() with different initial configurations
    using the `q_init` parameter.
    """

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        ee_site: str,
        joint_names: list[str] | None = None,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        collision_checker: CollisionChecker | None = None,
        damping: float = 0.1,
        max_iterations: int = 200,
        tolerance: float = 1e-3,
    ):
        """Initialize MuJoCo IK solver.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            ee_site: Name of end-effector site
            joint_names: List of joint names to control (if None, uses all joints)
            joint_limits: Optional (lower, upper) arrays for validation
            collision_checker: Optional collision checker for validation
            damping: Damping factor for damped least squares
            max_iterations: Maximum iterations for IK convergence
            tolerance: Position/orientation error tolerance for convergence
        """
        self.model = model
        self.data = data
        self.damping = damping
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.joint_limits = joint_limits
        self.collision_checker = collision_checker

        # Get site ID
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{ee_site}' not found in model")

        # Get joint indices and qpos addresses
        if joint_names is not None:
            self.joint_ids = []
            self.qpos_adrs = []
            self.dof_adrs = []
            for name in joint_names:
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jnt_id == -1:
                    raise ValueError(f"Joint '{name}' not found in model")
                self.joint_ids.append(jnt_id)
                self.qpos_adrs.append(model.jnt_qposadr[jnt_id])
                self.dof_adrs.append(model.jnt_dofadr[jnt_id])
            self.joint_ids = np.array(self.joint_ids)
            self.qpos_adrs = np.array(self.qpos_adrs)
            self.dof_adrs = np.array(self.dof_adrs)
        else:
            self.joint_ids = np.arange(model.njnt)
            self.qpos_adrs = np.array([model.jnt_qposadr[i] for i in range(model.njnt)])
            self.dof_adrs = np.array([model.jnt_dofadr[i] for i in range(model.njnt)])

        self._dof = len(self.joint_ids)

        # Get joint limits from model if not provided
        if self.joint_limits is None:
            # Check which joints are actually limited
            limited = self.model.jnt_limited[self.joint_ids]
            if np.any(limited):
                lower = self.model.jnt_range[self.joint_ids, 0].copy()
                upper = self.model.jnt_range[self.joint_ids, 1].copy()
                # For unlimited joints, use very large range
                lower[~limited.astype(bool)] = -1e10
                upper[~limited.astype(bool)] = 1e10
                self.joint_limits = (lower, upper)

        # Pre-allocate Jacobian arrays
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def _get_config(self) -> np.ndarray:
        """Get current joint configuration."""
        return np.array([self.data.qpos[adr] for adr in self.qpos_adrs])

    def _set_config(self, q: np.ndarray) -> None:
        """Set joint configuration."""
        for i, adr in enumerate(self.qpos_adrs):
            self.data.qpos[adr] = q[i]

    def _get_jacobian(self) -> np.ndarray:
        """Get 6xN Jacobian for the end-effector site."""
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.ee_site_id)
        # Extract columns for controlled joints only
        jacp = self._jacp[:, self.dof_adrs]
        jacr = self._jacr[:, self.dof_adrs]
        return np.vstack([jacp, jacr])

    def _pose_error(self, target_pose: np.ndarray) -> np.ndarray:
        """Compute 6D pose error (position + orientation)."""
        # Current pose
        current_pos = self.data.site_xpos[self.ee_site_id]
        current_rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3)

        # Target pose
        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]

        # Position error
        pos_error = target_pos - current_pos

        # Orientation error (using rotation matrix difference)
        rot_error_mat = target_rot @ current_rot.T
        # Convert to axis-angle
        rot_error = self._rotation_matrix_to_axis_angle(rot_error_mat)

        return np.concatenate([pos_error, rot_error])

    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle representation."""
        # Use MuJoCo's quat functions for robustness
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        # Convert quaternion to axis-angle
        angle = 2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))
        if angle < 1e-10:
            return np.zeros(3)
        axis = quat[1:4] / np.sin(angle / 2.0)
        return axis * angle

    def _clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp configuration to joint limits."""
        if self.joint_limits is not None:
            lower, upper = self.joint_limits
            return np.clip(q, lower, upper)
        return q

    def solve(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Solve IK for a single end-effector pose using differential IK.

        Args:
            pose: 4x4 homogeneous transform of desired end-effector pose
            q_init: Initial configuration (if None, uses current model state)

        Returns:
            List containing one solution if found, empty list otherwise
        """
        # Set initial configuration
        if q_init is not None:
            self._set_config(q_init)

        for _ in range(self.max_iterations):
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)

            # Compute error
            error = self._pose_error(pose)
            error_norm = np.linalg.norm(error)

            if error_norm < self.tolerance:
                # Converged
                q = self._get_config()
                return [q]

            # Get Jacobian
            J = self._get_jacobian()

            # Damped least squares: dq = J^T (J J^T + λ²I)^{-1} error
            JJT = J @ J.T
            damped = JJT + self.damping**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(damped, error)

            # Update configuration
            q = self._get_config() + dq
            q = self._clamp_to_limits(q)
            self._set_config(q)

        # Did not converge
        return []

    def solve_valid(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Solve IK and return only valid solutions.

        Filters solutions to return only those that are:
        - Within joint limits (if joint_limits provided)
        - Collision-free (if collision_checker provided)

        Args:
            pose: 4x4 homogeneous transform
            q_init: Initial configuration (if None, uses current model state)

        Returns:
            List of valid joint configurations (may be empty)
        """
        solutions = self.solve(pose, q_init)

        valid = []
        for q in solutions:
            # Check joint limits if provided
            if self.joint_limits is not None:
                lower, upper = self.joint_limits
                if not (np.all(q >= lower - 1e-6) and np.all(q <= upper + 1e-6)):
                    continue

            # Check collisions if checker provided
            if self.collision_checker is not None:
                if not self.collision_checker.is_valid(q):
                    continue

            valid.append(q)

        return valid

    def solve_from_multiple_inits(
        self,
        pose: np.ndarray,
        q_inits: list[np.ndarray],
        return_all: bool = False,
    ) -> list[np.ndarray]:
        """Solve IK from multiple initial configurations.

        Useful for finding multiple solutions with a differential IK solver.

        Args:
            pose: 4x4 homogeneous transform
            q_inits: List of initial configurations to try
            return_all: If True, return all solutions; if False, return first valid

        Returns:
            List of valid solutions
        """
        solutions = []
        for q_init in q_inits:
            result = self.solve_valid(pose, q_init)
            if result:
                solutions.extend(result)
                if not return_all:
                    return solutions
        return solutions
