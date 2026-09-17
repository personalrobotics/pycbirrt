# SPDX-License-Identifier: MIT
"""Native gafro bimanual backend: DualArmTaskSpace kinematics + IK.

Forward kinematics returns a :class:`~tsr.bimanual.BimanualPose` — the
(absolute, relative) pose pair a :class:`~tsr.bimanual.BimanualTSR` speaks in.
The IK solver (see :class:`GafroBimanualIKSolver`) stacks only the Jacobians of
the pose components that are present in the target.

Like :mod:`pycbirrt.backends.gafro` and :mod:`pycbirrt.backends.gafro_multiarm`,
this backend plans in the task space's **controlled** width: a gafro task space
reports a full chain DOF (the width its FK and Jacobians are evaluated at) and a
narrower controlled DOF (the width of its control Jacobian). ``dof``,
``joint_limits`` and every configuration crossing the planner/IK seam are the
controlled width; the non-controlled joints (a torso rail, say) are held at
:attr:`base_configuration` and scattered back in before FK.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gafro import DualArmTaskSpace, SystemSerialization
from tsr.bimanual import BimanualPose

from pycbirrt.backends.gafro import _as_vector, _controlled_joints, _extract_configuration

if TYPE_CHECKING:
    from gafro import System


def _resolve_task_space(system: "System", name: str, chain_names, control_groups):
    """The system's dual-arm task space called ``name``, or one built to order.

    A description that declares the task space itself (gafro YAML) is used as it
    stands. MJCF and URDF carry no task-space concept, so nothing of that name is
    registered after loading one, and the space is constructed from the system's
    two kinematic chains instead -- the same fallback
    :class:`~pycbirrt.backends.gafro_multiarm.GafroMultiArmModel` makes for three
    and four arms, which always build theirs.
    """
    if name in system.get_task_space_names():
        task_space = system.get_task_space(name)
        if not isinstance(task_space, DualArmTaskSpace):
            raise TypeError(
                f"task space {name!r} is a {type(task_space).__name__}, not a "
                f"DualArmTaskSpace; pass the name of a dual-arm task space, or "
                f"chain_names= to build one")
        return task_space

    chain_names = list(system.get_kinematic_chain_names() if chain_names is None
                       else chain_names)
    if len(chain_names) != 2:
        raise ValueError(
            f"a bimanual model needs exactly 2 kinematic chains, got "
            f"{len(chain_names)}: {chain_names} (use GafroMultiArmModel for three "
            f"or four)")

    return DualArmTaskSpace(system, name, chain_names, set(control_groups or ()))


class GafroBimanualModel:
    """Dual-arm kinematics over a gafro DualArmTaskSpace.

    Args:
        system: the gafro ``System``.
        task_space: name of the dual-arm task space. Used as it stands when the
            system already declares one by that name, otherwise the name given to
            a task space built from ``chain_names``.
        chain_names: the two kinematic chains to build the task space from, when
            the system does not already declare it. Defaults to the system's own
            chains, which must then be exactly two.
        control_groups: names of the System control groups that define which
            joints are planned. With none given gafro treats *every* joint in the
            chains as controlled (see
            :class:`~pycbirrt.backends.gafro.GafroRobotModel`). Ignored when the
            task space is already declared, since it carries its own.
        base_configuration: full task-width pose holding the non-controlled
            joints. Defaults to the task space's joint-limit midpoint.
    """

    def __init__(self, system: "System", task_space: str = "coop", chain_names=None,
                 control_groups=None, base_configuration: np.ndarray | None = None):
        self.system = system
        self.cooperative = _resolve_task_space(system, task_space, chain_names, control_groups)

        # Task-local indices of the controlled joints; these index into the
        # task-width configuration _extract_configuration returns, which is what
        # gafro's FK and Jacobians take.
        self._ctrl_idx = _controlled_joints(self.cooperative)

        # gafro returns joint limits as a JointPosition wrapper rather than an
        # array; _as_vector reads its coefficients (gafropy returned an array).
        system_lower = _as_vector(system.get_joint_limits_min())
        system_upper = _as_vector(system.get_joint_limits_max())
        full_lower = _extract_configuration(self.cooperative, system_lower)
        full_upper = _extract_configuration(self.cooperative, system_upper)
        self._lower = full_lower[self._ctrl_idx]
        self._upper = full_upper[self._ctrl_idx]

        if base_configuration is None:
            self._base_full = 0.5 * (full_lower + full_upper)
        else:
            self._base_full = np.asarray(base_configuration, dtype=float).copy()
            if self._base_full.shape != (self.cooperative.get_dof(),):
                raise ValueError(
                    f"base_configuration must be full task width "
                    f"({self.cooperative.get_dof()},), got {self._base_full.shape}")

        # Map each task-space joint to its index in the System's full config, so a
        # planned pose can be scattered back for visualization.
        ramp = np.arange(int(system.get_dof()), dtype=float)
        self._task_to_system = _extract_configuration(self.cooperative, ramp).astype(int)

        # The URDF/YAML-declared rest pose can sit fractions of a radian outside its
        # own declared limits (rounding in the robot description, e.g. gripper
        # joints on this rig) -- clamp so anything seeded from it is actually
        # valid, not silently poisoned from the start.
        self.default_system_configuration = np.clip(
            _as_vector(system.get_default_configuration()), system_lower, system_upper)

    @classmethod
    def from_file(cls, path: str, task_space: str = "coop", chain_names=None,
                  control_groups=None) -> "GafroBimanualModel":
        return cls(SystemSerialization.load(path), task_space, chain_names, control_groups)

    @property
    def dof(self) -> int:
        return len(self._ctrl_idx)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    @property
    def base_configuration(self) -> np.ndarray:
        """Full task-width config holding the non-controlled joints."""
        return self._base_full.copy()

    def _to_task_full(self, q: np.ndarray) -> np.ndarray:
        """Controlled-width ``q`` -> full task-width config (non-controlled held)."""
        q_full = self._base_full.copy()
        q_full[self._ctrl_idx] = np.asarray(q, dtype=float)
        return q_full

    def forward_kinematics(self, q: np.ndarray) -> BimanualPose:
        """The (absolute, relative) pose pair for a controlled-width ``q``.

        gafropy returned both task-space motors from one call; gafro exposes them
        as two accessors, each taking the full task-width configuration.
        """
        q_full = self._to_task_full(q)
        return BimanualPose(
            absolute=self.cooperative.compute_absolute_motor(q_full),
            relative=self.cooperative.compute_relative_motor(q_full),
        )

    def to_system_configuration(self, q: np.ndarray,
                                base: np.ndarray | None = None) -> np.ndarray:
        """Controlled-width ``q`` -> full System-width config for visualization.

        Scatters both arms' joints into their System indices on top of ``base``
        (the System default configuration if not given), so a ``Visualizer``
        driven by the System renders the planned pose while the rest of the robot
        (grippers, other chains) keeps ``base``'s pose.
        """
        if base is None:
            base = self.default_system_configuration
        system_q = np.asarray(base, dtype=float).copy()
        system_q[self._task_to_system] = self._to_task_full(q)
        return system_q

    def system_to_controlled(self, system_q: np.ndarray) -> np.ndarray:
        """System-width config -> controlled-width config.

        Inverse of :meth:`to_system_configuration` for the task space's joints;
        use it to seed planning from e.g. ``system.get_default_configuration()``.
        """
        return _extract_configuration(self.cooperative, system_q)[self._ctrl_idx]

    def normalize_pose(self, x) -> BimanualPose:
        """Pass-through: bimanual poses are already :class:`BimanualPose`."""
        if not isinstance(x, BimanualPose):
            raise TypeError(f"expected BimanualPose, got {type(x)}")
        return x


class GafroBimanualIKSolver:
    """Differential IK for a bimanual target, by damped least squares.

    The error stacks the CGA error twist(s) of whichever pose components are
    present in the target (absolute, relative, or both), and the step is the
    Levenberg-Marquardt solution against the matching stack of gafro's analytic
    geometric Jacobians -- the same scheme as
    :class:`~pycbirrt.backends.gafro.GafroIKSolver`, one arm wider.

    This used to hand the error and the Jacobian to SciPy's trust-region least
    squares (``scipy.optimize.least_squares``, ``"trf"``). That pairing is not
    sound: the error is ``log(current^-1 * target)`` while the Jacobian is the
    *twist* Jacobian, and the two differ by the exponential map's right
    Jacobian -- 40% apart at a tenth of a radian on this rig. A trust-region
    method takes that Jacobian as a prediction of the residual, finds the
    prediction wrong, and shrinks its step: it reached 1e-5 on 42% of random
    targets. Re-linearizing each step absorbs the difference instead, because
    the discrepancy is second order in the remaining error and vanishes as the
    error does. Measured over the same targets, it converges on all of them,
    two orders of magnitude tighter and faster.

    Configurations in and out are the model's controlled width.
    """

    def __init__(self, model: "GafroBimanualModel", collision_checker=None,
                 max_iterations: int = 200, tolerance: float = 1e-3,
                 damping: float = 0.1):
        self.model = model
        self.coop = model.cooperative
        self._ctrl_idx = model._ctrl_idx
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping
        self.collision_checker = collision_checker

    def _residual_and_jacobian(self, target: BimanualPose, q_full: np.ndarray):
        """Stacked error twist and Jacobian at a full task-width ``q_full``.

        The geometric Jacobians span the full task width, so only the controlled
        columns are kept -- that is the width of the search vector.
        """
        errors = []
        jacobians = []
        if target.absolute is not None:
            current = self.coop.compute_absolute_motor(q_full)
            errors.append(np.asarray(
                current.inverse().multiply(target.absolute).log(), dtype=float))
            jacobians.append(np.asarray(
                self.coop.compute_absolute_geometric_jacobian(q_full),
                dtype=float)[:, self._ctrl_idx])

        if target.relative is not None:
            current = self.coop.compute_relative_motor(q_full)
            errors.append(np.asarray(
                current.inverse().multiply(target.relative).log(), dtype=float))
            jacobians.append(np.asarray(
                self.coop.compute_relative_geometric_jacobian(q_full),
                dtype=float)[:, self._ctrl_idx])

        return np.concatenate(errors), np.vstack(jacobians)

    def solve(self, target: BimanualPose, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        if target.absolute is None and target.relative is None:
            raise ValueError("bimanual IK target has no active component")

        lower, upper = self.model.joint_limits
        if q_init is not None:
            q = np.asarray(q_init, dtype=float).copy()
        else:
            # The joint-limit midpoint is a poor seed for a coupled bimanual
            # target -- it's often near-singular for one arm or the other. The
            # model's declared rest pose is a real, reachable configuration, so it
            # converges far more reliably as a default.
            q = self.model.system_to_controlled(self.model.default_system_configuration)
        q = np.clip(q, lower, upper)

        for _ in range(self.max_iterations):
            error, J = self._residual_and_jacobian(target, self.model._to_task_full(q))
            if np.linalg.norm(error) < self.tolerance:
                return [q]

            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 error.
            JJT = J @ J.T
            damped = JJT + self.damping**2 * np.eye(JJT.shape[0])
            q = np.clip(q + J.T @ np.linalg.solve(damped, error), lower, upper)

        error, _ = self._residual_and_jacobian(target, self.model._to_task_full(q))
        return [q] if np.linalg.norm(error) < self.tolerance else []

    def solve_valid(self, target: BimanualPose, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        solutions = self.solve(target, q_init)
        valid = []
        lower, upper = self.model.joint_limits
        for q in solutions:
            if not (np.all(q >= lower - 1e-6) and np.all(q <= upper + 1e-6)):
                continue
            if self.collision_checker is not None and not self.collision_checker.is_valid(q):
                continue
            valid.append(q)
        return valid
