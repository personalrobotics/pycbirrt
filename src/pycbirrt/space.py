# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Joint-space geometry: limits, metric, interpolation, and uniform sampling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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

    @property
    def dof(self) -> int:
        return len(self.lower)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lower, self.upper

    def within_limits(self, q: np.ndarray) -> bool:
        """Whether ``q`` is inside the joint limits. Angular joints always pass."""
        q = np.asarray(q)
        inside = (q >= self.lower) & (q <= self.upper)
        if self.angular_joints is not None:
            inside = inside | self.angular_joints
        return bool(np.all(inside))

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
        """Uniform sample within the joint limits."""
        return rng.uniform(self.lower, self.upper)
