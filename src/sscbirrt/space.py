# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Joint-space geometry: limits, metric, interpolation, and uniform sampling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SpaceSampler(Protocol):
    """Proposes free-space targets for tree growth.

    ``JointSpace`` is the default (uniform within the limits, one full turn
    on angular joints). A replacement is a search strategy, not geometry:
    Gaussian sampling near obstacles, a restricted region, a deterministic
    sequence for tests. It must return configurations of shape ``(dof,)``;
    a target outside the space is handled by not growing toward it, so a
    sloppy sampler degrades rather than breaks. Replacing the default trades
    away probabilistic completeness unless the replacement has full support
    over the space; that is the caller's responsibility.
    """

    def sample(self, rng: np.random.Generator) -> np.ndarray: ...


class JointSpace:
    """A box-bounded joint space with optional angular (wraparound) joints.

    Owns everything the planner needs to know about the configuration space
    itself: joint limits, the distance metric, the direction between two
    configurations, straight-line interpolation, and uniform sampling.

    For angular joints the metric and direction wrap differences to
    ``[-pi, pi]`` and the limit check always passes. Interpolation is linear
    along the wrapped direction, which is correct for the small steps the
    planner takes.
    """

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        angular_joints: Sequence[bool] | None = None,
        joint_names: Sequence[str] | None = None,
    ):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape or self.lower.ndim != 1:
            raise ValueError("lower and upper must be 1-D arrays of the same length")
        if np.any(self.lower > self.upper):
            raise ValueError("lower limits must not exceed upper limits")

        self.angular_joints: np.ndarray | None = None
        if angular_joints is not None:
            if len(angular_joints) != self.dof:
                raise ValueError(f"angular_joints length ({len(angular_joints)}) must match robot DOF ({self.dof})")
            mask = np.asarray(angular_joints, dtype=bool)
            self.angular_joints = mask if mask.any() else None

        # Topology is the caller's declaration, never inferred. A bounded joint
        # needs finite limits (a sampler needs a bounded domain); an angular
        # joint has none, and whatever limits were stored for it are ignored.
        bounded = np.ones(self.dof, dtype=bool) if self.angular_joints is None else ~self.angular_joints
        finite = np.isfinite(self.lower) & np.isfinite(self.upper)
        bad = np.flatnonzero(bounded & ~finite)
        if bad.size:
            i = int(bad[0])
            name = f"joint '{joint_names[i]}' (index {i})" if joint_names is not None else f"joint {i}"
            raise ValueError(
                f"{name} has no finite limits [{self.lower[i]}, {self.upper[i]}]. If it turns continuously, declare "
                f"it with CBiRRTConfig(angular_joints=...); otherwise give the robot model finite planning limits "
                f"(for example MuJoCoRobotModel(..., joint_limits=(lower, upper)))"
            )
        # Sampling interval: the limits for bounded joints, one full turn for angular ones.
        self._sample_lower = self.lower.copy()
        self._sample_upper = self.upper.copy()
        if self.angular_joints is not None:
            self._sample_lower[self.angular_joints] = -np.pi
            self._sample_upper[self.angular_joints] = np.pi

    @property
    def dof(self) -> int:
        return len(self.lower)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lower, self.upper

    def within_limits(self, q: np.ndarray) -> bool:
        """Whether ``q`` is inside the joint limits. Angular joints always pass.

        Assumes ``q`` is a well-formed configuration; see ``contains`` for
        the full membership test.
        """
        q = np.asarray(q)
        inside = (q >= self.lower) & (q <= self.upper)
        if self.angular_joints is not None:
            inside = inside | self.angular_joints
        return bool(np.all(inside))

    def why_invalid(self, q) -> str | None:
        """Why ``q`` is not a member of this space, or None if it is.

        A member is a numeric array of shape ``(dof,)`` with finite entries,
        every bounded joint within its limits. Angular joints accept any
        finite value.
        """
        try:
            arr = np.asarray(q, dtype=float)
        except (TypeError, ValueError):
            return f"not numeric: {q!r}"
        if arr.shape != (self.dof,):
            return f"shape {arr.shape} != ({self.dof},)"
        if not np.all(np.isfinite(arr)):
            return f"non-finite entries: {arr}"
        if not self.within_limits(arr):
            bad = [i for i in range(self.dof) if not (self.lower[i] <= arr[i] <= self.upper[i])]
            if self.angular_joints is not None:
                bad = [i for i in bad if not self.angular_joints[i]]
            return "outside joint limits at " + ", ".join(
                f"joint {i}: {arr[i]:.4g} not in [{self.lower[i]:.4g}, {self.upper[i]:.4g}]" for i in bad
            )
        return None

    def contains(self, q) -> bool:
        """Whether ``q`` is a member of this space (see ``why_invalid``)."""
        return self.why_invalid(q) is None

    def direction(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        """Vector from ``q_from`` to ``q_to``, taking the short way around angular joints."""
        diff = np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float)
        if self.angular_joints is not None:
            a = self.angular_joints
            diff[a] = np.arctan2(np.sin(diff[a]), np.cos(diff[a]))
        return diff

    def distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Euclidean norm of ``direction(q1, q2)``."""
        return float(np.linalg.norm(self.direction(q1, q2)))

    def interpolate(self, q_from: np.ndarray, q_to: np.ndarray, t: float) -> np.ndarray:
        """Point a fraction ``t`` along the straight line from ``q_from`` toward ``q_to``."""
        return np.asarray(q_from, dtype=float) + t * self.direction(q_from, q_to)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Uniform sample: within the limits on bounded joints, over one full turn on angular joints."""
        return rng.uniform(self._sample_lower, self._sample_upper)

    def unwrap_path(self, path: list[np.ndarray]) -> list[np.ndarray]:
        """Re-express a path so consecutive raw values on angular joints take the short way.

        Walks forward from the first waypoint, which is returned as given,
        setting ``q[i] = q[i-1] + direction(q[i-1], q[i])``. Each waypoint
        stays the same physical configuration (angular joints are periodic and
        accept any finite value), but an executor interpolating raw joint
        values no longer sees a full-turn jump where two tree nodes were
        stored in representations one turn apart. Non-angular joints are
        untouched, and a space with no angular joints returns the path
        unchanged. The last waypoint may therefore differ from the goal as
        given by a multiple of 2π on an angular joint (#77).
        """
        if self.angular_joints is None or len(path) < 2:
            return list(path)
        out = [np.array(path[0], dtype=float)]
        for q in path[1:]:
            q = np.asarray(q, dtype=float)
            nxt = out[-1] + self.direction(out[-1], q)
            nxt[~self.angular_joints] = q[~self.angular_joints]  # non-angular joints copied exactly
            out.append(nxt)
        return out
