"""MuJoCo backend for robot model and collision checking."""

import numpy as np

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
