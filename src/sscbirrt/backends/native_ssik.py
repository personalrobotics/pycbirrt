# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Render an ``ssik.Manipulator`` into the native SSIK adapter (docs/native-design.md, v1.6.0 addendum)."""

from __future__ import annotations

import numpy as np

from sscbirrt import _native

VERIFIED_FAMILIES = ("ikgeo.three_parallel",)


def unsupported_reason(manipulator) -> str | None:
    """Why this manipulator cannot be lifted natively, or None."""
    if not _native.has_ssik():
        return _native.ssik_unavailable_reason()
    name = getattr(manipulator, "solver_name", None)
    if name not in VERIFIED_FAMILIES:
        return f"SSIK family {name} is not in the native allowlist (verified: {', '.join(VERIFIED_FAMILIES)})"
    if getattr(manipulator, "dof", None) != 6:
        return f"native SSIK support needs a 6-DOF arm, got {getattr(manipulator, 'dof', None)}"
    return None


def arm_from_manipulator(manipulator, *, T_base: np.ndarray | None = None, T_ee: np.ndarray | None = None):
    """``ssik.cpp.joint_data(manipulator)`` handed to ``_native.SSIKArm`` with the adapter's frame offsets."""
    import ssik.cpp

    why = unsupported_reason(manipulator)
    if why is not None:
        raise ValueError(why)
    d = ssik.cpp.joint_data(manipulator)
    eye = np.eye(4)
    return _native.SSIKArm(
        d.solver,
        [list(map(float, a)) for a in d.axis],
        [list(map(float, np.asarray(t, dtype=float).reshape(16))) for t in d.t_left],
        [list(map(float, np.asarray(t, dtype=float).reshape(16))) for t in d.t_right],
        [int(t) for t in d.joint_type],
        [float(v) for v in d.lo],
        [float(v) for v in d.hi],
        [bool(v) for v in d.present],
        (eye if T_base is None else np.asarray(T_base, dtype=float)).tolist(),
        (eye if T_ee is None else np.asarray(T_ee, dtype=float)).tolist(),
    )
