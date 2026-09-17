# SPDX-License-Identifier: MIT
"""Native gafro backend for three- and four-arm cooperative systems.

A dual-arm system is described by an (absolute, relative) Motor pair; a three-
or four-arm system is described instead by a single
:class:`~gafro.SimilarityTransformation` — the transform carrying a canonical
primitive onto the one the end-effectors span (a **circle** for three arms, a
**sphere** for four). That transform carries a dilation as well as a rotation
and translation, because the arms can grow or shrink the shape they hold.

Forward kinematics here therefore returns a ``SimilarityTransformation``, which
is exactly what :class:`~tsr.multiarm.CircleTSR` and
:class:`~tsr.multiarm.SphereTSR` speak in.

**On the Jacobian.** gafro exposes ``compute_geometric_jacobian`` (7 rows) for
these task spaces, but its rows are in the similarity's own bivector basis, not
the (translation, rotation, dilation) canonical-decomposition coordinates the
TSRs are defined over — and for a four-arm (sphere) space its three rotation
rows are identically zero, since a sphere has no rotational freedom. Rather than
assume a mapping between the two bases, the IK solver below differentiates the
TSR coordinates directly. That keeps the Jacobian and the error in the *same*
coordinates by construction, which is what makes the damped least-squares step
well-posed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gafro import QuadrupleCooperativeTaskSpace, SystemSerialization, TripleCooperativeTaskSpace
from tsr.multiarm import CircleTSR, SphereTSR, is_degenerate

from pycbirrt.backends.gafro import (
    _controlled_joints,
    _extract_configuration,
)

if TYPE_CHECKING:
    from gafro import System

# Arm count -> (gafro task space, matching TSR class).
_TASK_SPACES = {
    3: (TripleCooperativeTaskSpace, CircleTSR),
    4: (QuadrupleCooperativeTaskSpace, SphereTSR),
}


class GafroMultiArmModel:
    """Three- or four-arm kinematics over a gafro cooperative task space.

    Args:
        system: the gafro ``System``.
        chain_names: one kinematic chain per arm; 3 or 4 of them. The count
            selects the task space (circle or sphere).
        control_groups: names of the System control groups that define which
            joints are planned. With none given gafro treats *every* joint in
            the chains as controlled (see :class:`GafroRobotModel`).
        name: task-space name (default ``"coop"``).
    """

    def __init__(self, system: "System", chain_names, control_groups=None, name: str = "coop"):
        chain_names = list(chain_names)
        if len(chain_names) not in _TASK_SPACES:
            raise ValueError(
                f"multi-arm model needs 3 or 4 chains, got {len(chain_names)}: "
                f"{chain_names} (use GafroBimanualModel for two)")
        task_space_cls, tsr_cls = _TASK_SPACES[len(chain_names)]
        self.system = system
        self.chain_names = chain_names
        self.control_groups = set(control_groups or ())
        self.tsr_class = tsr_cls
        self.cooperative = task_space_cls(system, name, chain_names, self.control_groups)

        self._ctrl_idx = _controlled_joints(self.cooperative)
        full_lower = _extract_configuration(self.cooperative, system.get_joint_limits_min())
        full_upper = _extract_configuration(self.cooperative, system.get_joint_limits_max())
        self._lower = full_lower[self._ctrl_idx]
        self._upper = full_upper[self._ctrl_idx]
        self._base_full = 0.5 * (full_lower + full_upper)
        self.default_system_configuration = np.asarray(
            system.get_default_configuration(), dtype=float)

    @classmethod
    def from_file(cls, path: str, chain_names, control_groups=None, name: str = "coop"):
        return cls(SystemSerialization.load(path), chain_names, control_groups, name)

    @property
    def arm_count(self) -> int:
        return len(self.chain_names)

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

    def forward_kinematics(self, q: np.ndarray):
        """Task pose as a ``SimilarityTransformation``."""
        return self.cooperative.compute_similarity_transformation(self._to_task_full(q))

    def normalize_pose(self, x):
        """Pass-through: multi-arm poses are already similarity transforms."""
        return x

    def is_degenerate_configuration(self, q: np.ndarray) -> bool:
        """Whether ``q`` puts the end-effectors in a pose with no finite scale.

        A sampling planner will visit such configurations (four coplanar
        end-effectors have no circumsphere); they are unusable rather than
        erroneous, so callers should skip them.
        """
        return is_degenerate(self.forward_kinematics(q))


class GafroMultiArmIKSolver:
    """Damped least-squares IK for a three- or four-arm cooperative target.

    The error and the Jacobian are both expressed in the target TSR's own
    coordinates (``[tx, ty, tz, dilation]`` for a sphere;
    ``[tx, ty, tz, dilation, n1, n2]`` for a circle), so the least-squares step
    is well-posed without any assumption about gafro's internal Jacobian basis.
    The Jacobian is obtained by central differences on those coordinates.
    """

    def __init__(self, model: "GafroMultiArmModel", collision_checker=None,
                 damping: float = 0.1, max_iterations: int = 200, tolerance: float = 1e-3,
                 step: float = 1e-6, escape_attempts: int = 8, escape_scale: float = 0.15,
                 seed: int = 0):
        self.model = model
        self.escape_attempts = escape_attempts
        self.escape_scale = escape_scale
        self._rng = np.random.default_rng(seed)
        self.damping = damping
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.step = step
        self.collision_checker = collision_checker

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        lower, upper = self.model.joint_limits
        return np.clip(q, lower, upper)

    def _coordinates(self, tsr, q: np.ndarray):
        """TSR coordinates at ``q``, or None where the primitive degenerates."""
        pose = self.model.forward_kinematics(q)
        if is_degenerate(pose):
            return None
        return tsr.to_bw(pose)

    def _jacobian(self, tsr, q: np.ndarray) -> np.ndarray:
        """Central-difference Jacobian of the TSR coordinates w.r.t. ``q``."""
        rows = tsr.dof
        jacobian = np.zeros((rows, len(q)))
        for i in range(len(q)):
            step = np.zeros(len(q))
            step[i] = self.step
            forward = self._coordinates(tsr, q + step)
            backward = self._coordinates(tsr, q - step)
            if forward is None or backward is None:
                return None
            jacobian[:, i] = (forward - backward) / (2.0 * self.step)
        return jacobian

    def solve(self, target, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Drive the task pose into ``target``.

        Args:
            target: a :class:`~tsr.multiarm.CircleTSR` / ``SphereTSR`` (the
                solver aims at the closest point of the region), or a bare
                ``SimilarityTransformation`` to hit exactly.
            q_init: controlled-width seed; defaults to the limit midpoint.

        Returns:
            ``[q]`` on convergence, else ``[]``.
        """
        tsr, goal_bw = self._resolve_target(target)

        lower, upper = self.model.joint_limits
        q = (np.asarray(q_init, dtype=float).copy()
             if q_init is not None else 0.5 * (lower + upper))
        # A perfectly symmetric seed (all joints equal -- the limit midpoint is
        # one) puts the end-effectors on the degenerate surface, where the
        # coordinates have no finite scale and the Jacobian cannot be formed.
        # Jitter off it before starting rather than declaring failure.
        q = self._escape_degeneracy(tsr, q)
        if q is None:
            return []

        for _ in range(self.max_iterations):
            current = self._coordinates(tsr, q)
            if current is None:
                return []  # degenerate: no usable pose here
            error = self._error(tsr, current, goal_bw)
            if np.linalg.norm(error) < self.tolerance:
                return [q]

            J = self._jacobian(tsr, q)
            if J is None:
                return []
            JJT = J @ J.T
            damped = JJT + self.damping**2 * np.eye(JJT.shape[0])
            q = self._clamp(q + J.T @ np.linalg.solve(damped, error))

        return []

    def _escape_degeneracy(self, tsr, q: np.ndarray):
        """Nudge ``q`` until the pose and its Jacobian are both well defined.

        Returns the (possibly jittered) configuration, or None if no nearby
        non-degenerate configuration was found.
        """
        lower, upper = self.model.joint_limits
        candidate = q
        for attempt in range(self.escape_attempts + 1):
            if (self._coordinates(tsr, candidate) is not None
                    and self._jacobian(tsr, candidate) is not None):
                return candidate
            # Grow the perturbation so a badly degenerate seed still escapes.
            scale = self.escape_scale * (attempt + 1)
            candidate = np.clip(q + self._rng.normal(0.0, scale, size=len(q)), lower, upper)
        return None

    def _resolve_target(self, target):
        """Split a target into (region, fixed goal coordinates or None)."""
        if isinstance(target, (CircleTSR, SphereTSR)):
            return target, None
        # A bare similarity transform: aim at its exact coordinates, using a
        # zero-width region of the right shape to read them off.
        tsr = self.model.tsr_class(Bw=np.zeros((self.model.tsr_class._DOF, 2)))
        if is_degenerate(target):
            raise ValueError(
                "multi-arm IK target has no finite scale (degenerate primitive); "
                "it cannot be reached by any configuration")
        return tsr, tsr.to_bw(target)

    @staticmethod
    def _error(tsr, current: np.ndarray, goal_bw):
        """Coordinate error: to a fixed goal, or to the nearest point of a region."""
        if goal_bw is not None:
            return goal_bw - current
        box = tsr._Bw_cont
        return np.clip(current, box[:, 0], box[:, 1]) - current

    def solve_valid(self, target, q_init: np.ndarray | None = None) -> list[np.ndarray]:
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
