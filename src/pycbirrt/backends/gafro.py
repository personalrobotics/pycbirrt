# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Native gafro backend for robot kinematics and differential IK.

This backend pushes all geometry through Conformal Geometric Algebra: forward
kinematics returns a ``Motor`` directly (no matrix round-trip), and the
differential IK solver expresses its pose error as the **world-frame Motor
bivector log** — the exact quantity the gafro geometric Jacobian integrates
(verified: ``J @ dq == log(M_new * M_old^-1)``). There are no rotation
matrices, quaternions, or Euler angles anywhere in this backend.

gafro has no collision engine, so collision checking is delegated to an
injected :class:`~pycbirrt.interfaces.collision_checker.CollisionChecker`
(e.g. ``MuJoCoCollisionChecker``).
"""

from typing import TYPE_CHECKING

import numpy as np
from gafro import SingleArmTaskSpace, SystemSerialization

from pycbirrt.interfaces.collision_checker import CollisionChecker

if TYPE_CHECKING:
    from gafro import Motor, System


def as_motor(x) -> "Motor":
    """Coerce a pose to a gafro ``Motor``.

    gafropy's ``Motor(x)`` constructor was polymorphic (Motor / 4x4 matrix);
    gafro's takes no such argument, so the coercion is explicit here and shared
    by every backend that accepts "a Motor or a 4x4".
    """
    from gafro import Motor

    if isinstance(x, Motor):
        return x
    return Motor.from_matrix(np.asarray(x, dtype=float))


def _as_vector(value) -> np.ndarray:
    """Numeric vector from a gafro joint quantity or a plain array.

    System-level joint limits come back as a ``JointPosition`` wrapper rather
    than an array; its ``coefficients()`` are the numbers. Plain arrays (e.g.
    ``get_default_configuration()``) pass straight through.
    """
    coefficients = getattr(value, "coefficients", None)
    if callable(coefficients):
        value = coefficients()
    return np.asarray(value, dtype=float).ravel()


def _extract_configuration(task_space, system_vector) -> np.ndarray:
    """Gather a System-width vector down to ``task_space``'s chain width.

    gafropy's task spaces had an ``extract_configuration`` method; gafro instead
    exposes the mapping as ``get_joint_indices()`` -- the System-width index of
    each of the task space's joints, in task order -- so the gather is an
    explicit fancy-index. (Do not confuse it with
    ``get_task_space_joint_indices()``, which is task-local: for a dual-arm
    space those are 0..13 while the System indices skip the gripper joints.)
    """
    idx = np.asarray(task_space.get_joint_indices(), dtype=int)
    return _as_vector(system_vector)[idx]


def _controlled_joints(task_space) -> np.ndarray:
    """Task-local indices of the controlled joints (gafropy: get_controlled_joints).

    These index into the *task-width* configuration vector, which is what
    ``_extract_configuration`` returns.
    """
    return np.asarray(task_space.get_task_space_joint_indices(), dtype=int)


def _ee_error_twist(target: "Motor", current: "Motor") -> np.ndarray:
    """6-vector world-frame error twist from ``current`` toward ``target``.

    Equals ``log(target * current^-1)``; this is the twist the gafro
    geometric Jacobian maps joint velocities onto, so ``J @ dq`` drives this
    error to zero. Order matches ``Motor.log()``:
    ``[e12, e13, e23, e1i, e2i, e3i]`` (rotation bivector, then translation).
    """
    return np.asarray(target.multiply(current.inverse()).log(), dtype=float)


class GafroRobotModel:
    """Robot model backed by a gafro :class:`System` (FK + joint limits).

    The end-effector pose and geometric Jacobian are exposed through a
    :class:`~gafro.SingleArmTaskSpace` built over one kinematic chain of the
    system; the underlying ``System`` is kept on :attr:`system` so callers can,
    e.g., drive a ``Visualizer`` from the same model without reloading it.
    """

    def __init__(self, system: "System", chain_name: str | None = None,
                 control_groups: "set[str] | None" = None):
        if chain_name is None:
            chain_names = system.get_kinematic_chain_names()
            if not chain_names:
                raise ValueError("System has no kinematic chains")
            chain_name = chain_names[-1]
        self.system = system
        self.chain_name = chain_name
        self.control_groups = set(control_groups or ())
        self.manipulator = SingleArmTaskSpace(system, chain_name, chain_name,
                                              self.control_groups)
        # A task space reports a full chain DOF (configs / limits / FK input) and a
        # narrower *controlled* DOF (the width of its control Jacobian). Planning
        # happens in the controlled width; the non-controlled joints (e.g. a
        # prismatic torso) are held at the limit midpoint for FK / visualization.
        #
        # The split comes from the task space's control groups. With none given,
        # gafro treats *every* joint in the chain as controlled -- so a chain that
        # runs through a torso rail will plan the torso too. Formats that carry no
        # control-group concept (MJCF, URDF) import none, so pass ``control_groups``
        # (see :meth:`System.add_control_group`) to get the intended partition.
        self._ctrl_idx = _controlled_joints(self.manipulator)
        # Joint limits live on the System now (a task space no longer carries its
        # own). extract_configuration gathers this chain's joints out of the
        # System-width limit vectors, giving the full task-width limits _ctrl_idx
        # indexes into -- same system->task mapping used for _task_to_system below.
        full_lower = _extract_configuration(self.manipulator, system.get_joint_limits_min())
        full_upper = _extract_configuration(self.manipulator, system.get_joint_limits_max())
        self._lower = full_lower[self._ctrl_idx]
        self._upper = full_upper[self._ctrl_idx]
        self._base_full = 0.5 * (full_lower + full_upper)
        # Map each task-space joint to its index in the System's full config. The
        # task space's extract_configuration() does system -> task; feeding it a
        # ramp recovers, for each task joint k, the system index it came from.
        self._system_dof = int(system.get_dof())
        ramp = np.arange(self._system_dof, dtype=float)
        self._task_to_system = _extract_configuration(self.manipulator, ramp).astype(int)
        # The System's default pose for the whole robot; used as the base for
        # visualization so non-chain joints (other arms, torso) keep their pose.
        self.default_system_configuration = np.asarray(
            system.get_default_configuration(), dtype=float)

    @classmethod
    def from_file(cls, path: str, chain_name: str | None = None,
                  control_groups: "set[str] | None" = None) -> "GafroRobotModel":
        """Load a robot description (URDF / MJCF / YAML) and wrap one of its chains.

        If ``chain_name`` is omitted, the system's last kinematic chain is used.
        """
        return cls(SystemSerialization.load(path), chain_name, control_groups)

    # Back-compat alias; ``from_file`` is the generic name (loads URDF or MJCF).
    from_urdf = from_file

    @property
    def dof(self) -> int:
        return len(self._ctrl_idx)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    @property
    def base_configuration(self) -> np.ndarray:
        """Full chain-width config holding the non-controlled joints.

        Hand this to :class:`GafroIKSolver` so the solver poses the chain's
        non-controlled joints (e.g. a torso rail) exactly as this model's FK
        does; otherwise IK searches a differently-posed arm and targets produced
        by :meth:`forward_kinematics` come out unreachable.
        """
        return self._base_full.copy()

    def normalize_pose(self, x) -> "Motor":
        """Coerce a forward-kinematics / TSR pose to a ``Motor``.

        This is the single-arm pose token. It moves the ``Motor(...)`` coercion
        the planner used to do inline into the model, so the planner can stay
        pose-shape-agnostic (a bimanual model returns a BimanualPose instead).
        """
        return as_motor(x)

    def forward_kinematics(self, q: np.ndarray) -> "Motor":
        """End-effector pose as a ``gafro.Motor`` (no matrix round-trip).

        ``q`` is controlled-width (:attr:`dof`); it is scattered into the full
        chain config (non-controlled joints held at the limit midpoint) before FK.
        """
        return self.manipulator.compute_ee_motor(self._to_task_full(q))

    def _to_task_full(self, q: np.ndarray) -> np.ndarray:
        """Controlled-width ``q`` -> full task-width config (non-controlled held)."""
        q_full = self._base_full.copy()
        q_full[self._ctrl_idx] = np.asarray(q, dtype=float)
        return q_full

    def to_system_configuration(self, q: np.ndarray,
                                base: np.ndarray | None = None) -> np.ndarray:
        """Controlled-width ``q`` -> full System-width config for visualization.

        Scatters this chain's joints into their System indices on top of ``base``
        (the System default configuration if not given), so a ``Visualizer``
        driven by the System renders this arm in the planned pose while the rest
        of the robot (other arms, torso) keeps ``base``'s pose. Front-padding, by
        contrast, mis-assigns the joint slots and collapses the whole robot.
        """
        if base is None:
            base = self.default_system_configuration
        system_q = np.asarray(base, dtype=float).copy()
        system_q[self._task_to_system] = self._to_task_full(q)
        return system_q

    def system_to_controlled(self, system_q: np.ndarray) -> np.ndarray:
        """System-width config -> this chain's controlled-width config.

        Inverse of :meth:`to_system_configuration` for the chain's joints; use it
        to seed planning from e.g. ``system.get_default_configuration()``.
        """
        task = _extract_configuration(self.manipulator, system_q)
        return task[self._ctrl_idx]


class GafroIKSolver:
    """Differential IK using the gafro geometric Jacobian and CGA pose error.

    Damped least squares (Levenberg-Marquardt) in the same spirit as the MuJoCo
    differential solver, but the error and Jacobian are CGA-native. Returns at
    most one solution per ``solve`` call (found iteratively from ``q_init``); to
    enumerate solutions, call from multiple initial configurations.
    """

    def __init__(
        self,
        manipulator: "SingleArmTaskSpace",
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        collision_checker: CollisionChecker | None = None,
        damping: float = 0.1,
        max_iterations: int = 200,
        tolerance: float = 1e-3,
        system: "System | None" = None,
        base_configuration: np.ndarray | None = None,
    ):
        self.manipulator = manipulator
        self.damping = damping
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.collision_checker = collision_checker

        # IK searches in the task space's *controlled* width (the column count of
        # its control Jacobian); the non-controlled joints are held fixed. FK and
        # the Jacobian are evaluated on the full chain config, so we scatter the
        # controlled vector into a full-width base via the controlled-joint index.
        self._ctrl_idx = _controlled_joints(manipulator)
        self._dof = manipulator.get_controlled_dof()
        # Joint limits are System-level; _extract_configuration maps the
        # System-width limit vectors down to this task space's full chain width
        # (what _ctrl_idx indexes into). gafro task spaces do not carry a
        # back-reference to their System (gafropy's did, via get_system()), so
        # the System must be supplied whenever the limits have to be derived.
        if system is not None:
            full_lower = _extract_configuration(manipulator, system.get_joint_limits_min())
            full_upper = _extract_configuration(manipulator, system.get_joint_limits_max())
            self._mid_full = 0.5 * (full_lower + full_upper)
            if joint_limits is None:
                joint_limits = (full_lower[self._ctrl_idx], full_upper[self._ctrl_idx])
        elif base_configuration is not None:
            self._mid_full = np.asarray(base_configuration, dtype=float).copy()
            if self._mid_full.shape != (manipulator.get_dof(),):
                raise ValueError(
                    f"base_configuration must be full chain width "
                    f"({manipulator.get_dof()},), got {self._mid_full.shape}")
        else:
            # Without the System (or an explicit base) there is nothing to hold the
            # non-controlled joints at. Guessing zero silently moves them, which
            # puts FK on a differently-posed arm than the caller's -- so refuse.
            raise ValueError(
                "GafroIKSolver needs system= (or base_configuration=) to know where "
                "to hold the chain's non-controlled joints; gafro task spaces have "
                "no get_system()")
        self.joint_limits = joint_limits

    def _clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        lower, upper = self.joint_limits
        return np.clip(q, lower, upper)

    def solve(self, pose: "Motor | np.ndarray", q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Solve IK for a single end-effector pose using differential IK.

        Args:
            pose: Desired end-effector pose as a ``gafro.Motor`` (a 4x4
                homogeneous transform / 6-log is also accepted via ``Motor``).
            q_init: Initial configuration. May be controlled-width (``_dof``) or
                full chain width; a full-width hint also seeds the held values of
                the non-controlled joints (so e.g. the torso stays where the
                caller put it). If None, starts from the controlled-limit midpoint
                with non-controlled joints at the full-limit midpoint.

        Returns:
            List containing one controlled-width solution if converged, else [].
        """
        target = as_motor(pose)

        # Full-width base carrying the held (non-controlled) joint values, and the
        # controlled-width search vector q.
        base_full = self._mid_full.copy()
        if q_init is not None:
            q_init = np.asarray(q_init, dtype=float)
            if q_init.shape[0] == self.manipulator.get_dof():
                base_full = q_init.copy()
                q = q_init[self._ctrl_idx].copy()
            else:
                q = q_init.copy()
        else:
            lower, upper = self.joint_limits
            q = 0.5 * (lower + upper)

        for _ in range(self.max_iterations):
            q_full = base_full.copy()
            q_full[self._ctrl_idx] = q
            current = self.manipulator.compute_ee_motor(q_full)
            error = _ee_error_twist(target, current)
            if np.linalg.norm(error) < self.tolerance:
                return [q]

            # The geometric Jacobian spans the full chain width; keep only the
            # controlled columns so dq matches the controlled-width search vector.
            J = np.asarray(self.manipulator.compute_geometric_jacobian(q_full), dtype=float)
            J = J[:, self._ctrl_idx]
            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 error
            JJT = J @ J.T
            damped = JJT + self.damping**2 * np.eye(JJT.shape[0])
            dq = J.T @ np.linalg.solve(damped, error)

            q = self._clamp_to_limits(q + dq)

        return []

    def solve_valid(self, pose: "Motor | np.ndarray", q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Solve IK and return only solutions within limits and collision-free."""
        solutions = self.solve(pose, q_init)

        valid = []
        lower, upper = self.joint_limits
        for q in solutions:
            if not (np.all(q >= lower - 1e-6) and np.all(q <= upper + 1e-6)):
                continue
            if self.collision_checker is not None and not self.collision_checker.is_valid(q):
                continue
            valid.append(q)
        return valid

    def solve_from_multiple_inits(
        self,
        pose: "Motor | np.ndarray",
        q_inits: list[np.ndarray],
        return_all: bool = False,
    ) -> list[np.ndarray]:
        """Solve IK from multiple initial configurations (to find >1 solution)."""
        solutions: list[np.ndarray] = []
        for q_init in q_inits:
            result = self.solve_valid(pose, q_init)
            if result:
                solutions.extend(result)
                if not return_all:
                    return solutions
        return solutions
