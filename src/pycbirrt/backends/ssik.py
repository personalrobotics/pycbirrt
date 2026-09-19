# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""SSIK backend: enumerative analytical inverse kinematics.

SSIK (``pip install "pycbirrt[ssik]"``) solves 6R and 7R manipulators in
closed form, accepts a seed, and, from version 6, returns every in-limit
winding of each geometric branch for joints whose range exceeds one turn.
That is exactly what ``TSRConfigurationSet`` needs to represent the complete
TSR-induced configuration set on robots like the UR5e (#36, #63).

This adapter is deliberately thin. It wraps any SSIK solver object, an
``ssik.Manipulator`` or a prebuilt artifact module such as
``ssik.prebuilt.ur5e_ik``, and exposes pycbirrt's ``IKSolver.solve``. It does
no collision checking and applies no solution cap. Diagnostics, refinement,
tolerance policies, and robot construction stay on the SSIK object; configure
it before wrapping.

Frame and joint-order contract
------------------------------
The wrapped SSIK model and pycbirrt's ``RobotModel`` must agree on joint
order and sign, on the base frame, on the end-effector frame, and on which
joints are finite versus continuous. The adapter does not guess. If the SSIK
model's frames differ from the robot model's by fixed transforms, pass them
explicitly: ``T_base`` maps SSIK's base frame into the robot model's base
frame and ``T_ee`` maps SSIK's end-effector frame to the robot model's
end-effector frame, so that ``robot.forward_kinematics(q) == T_base @ ssik.fk(q) @ T_ee``.
``SSIKSolver.fk`` applies the same transforms, so you can assert that
equality in a test before planning; ``tests/test_ur5e_integration.py`` does.

For a MuJoCo model, ``ssik.Manipulator.from_mjcf(xml, base="world", ee=<body>)``
with ``T_ee`` set to the end-effector site's offset within that body (see
``pycbirrt.backends.mujoco.site_offset_in_body``) reproduces MuJoCo's forward
kinematics to machine precision. Prebuilt SSIK artifacts use the vendor's
nominal geometry and may differ from a simulator model by a millimeter; check
before relying on tight membership tolerances.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

try:
    import ssik  # noqa: F401
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'The SSIK backend requires the optional dependency ssik>=6.0.1. Install it with: pip install "pycbirrt[ssik]"'
    ) from e


class _SSIKLike(Protocol):
    def solve(self, T_target: Any, **kwargs: Any) -> list[Any]: ...
    def fk(self, q: Any) -> np.ndarray: ...


class SSIKSolver:
    """Adapt an SSIK solver to pycbirrt's ``IKSolver`` protocol.

    Args:
        solver: An ``ssik.Manipulator`` or a prebuilt artifact module, anything
            with ``solve(T, q_seed=..., respect_limits=..., enumerate_windings=...)``
            returning objects with a ``q`` attribute, and ``fk(q)``.
        T_base: Optional 4x4 transform from SSIK's base frame to the robot
            model's base frame. Copied at construction; fixed thereafter.
        T_ee: Optional 4x4 transform from SSIK's end-effector frame to the
            robot model's end-effector frame. Copied at construction; fixed
            thereafter.

    ``solve`` requests limit-respecting solutions with winding enumeration on
    (SSIK's default), forwards ``q_init`` as ``q_seed``, imposes no solution
    cap, and returns each solution's ``q`` as an independent float array.
    """

    def __init__(self, solver: _SSIKLike, *, T_base: np.ndarray | None = None, T_ee: np.ndarray | None = None):
        if not callable(getattr(solver, "solve", None)):
            raise TypeError(f"SSIKSolver expects an SSIK solver with a solve() method, got {type(solver).__name__}")
        self.solver = solver
        self.T_base = None if T_base is None else _check_transform(T_base, "T_base")
        self.T_ee = None if T_ee is None else _check_transform(T_ee, "T_ee")
        self._T_base_inv = None if self.T_base is None else np.linalg.inv(self.T_base)
        self._T_ee_inv = None if self.T_ee is None else np.linalg.inv(self.T_ee)

    def solve(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        target = np.asarray(pose, dtype=float)
        if target.shape != (4, 4):
            raise ValueError(f"pose must be a 4x4 transform, got shape {target.shape}")
        if self._T_base_inv is not None:
            target = self._T_base_inv @ target
        if self._T_ee_inv is not None:
            target = target @ self._T_ee_inv

        kwargs: dict[str, Any] = {"respect_limits": True, "enumerate_windings": True}
        if q_init is not None:
            kwargs["q_seed"] = np.asarray(q_init, dtype=float)
        solutions = self.solver.solve(target, **kwargs)

        out: list[np.ndarray] = []
        for i, sol in enumerate(solutions):
            q = np.array(getattr(sol, "q", sol), dtype=float)  # independent copy
            if q.ndim != 1 or not np.all(np.isfinite(q)):
                raise ValueError(f"SSIK returned a malformed configuration at index {i}: {q!r}")
            out.append(q)
        return out

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics in the robot model's frames (``T_base @ ssik.fk(q) @ T_ee``)."""
        T = np.asarray(self.solver.fk(np.asarray(q, dtype=float)), dtype=float)
        if self.T_base is not None:
            T = self.T_base @ T
        if self.T_ee is not None:
            T = T @ self.T_ee
        return T


def _check_transform(T: np.ndarray, name: str) -> np.ndarray:
    """Validate and take an independent, read-only copy of a fixed frame transform.

    The adapter caches the inverse at construction; owning the array means a
    caller mutating their original can never make the stored transform and
    its cached inverse disagree (#66).
    """
    T = np.array(T, dtype=float, copy=True)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError(f"{name} must be a finite 4x4 transform, got shape {T.shape}")
    T.setflags(write=False)
    return T
