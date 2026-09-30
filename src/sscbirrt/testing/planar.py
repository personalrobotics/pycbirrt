# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""A two-link planar arm with closed-form IK: the reference robot for sscbirrt.

Link lengths 1 and 1, revolute joints limited to [-pi, pi], end effector at
the tip with identity orientation. Every operation is analytic and
deterministic, which is what a behavior artifact needs.
"""

from __future__ import annotations

import numpy as np


class PlanarArm:
    """Two-link planar arm; ``RobotModel`` protocol."""

    def __init__(self, l1: float = 1.0, l2: float = 1.0):
        self.l1 = l1
        self.l2 = l2

    @property
    def dof(self) -> int:
        return 2

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi])

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        x = self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1])
        y = self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1])
        T = np.eye(4)
        T[0, 3] = x
        T[1, 3] = y
        return T


class PlanarIK:
    """Closed-form IK for ``PlanarArm``; ``IKSolver`` protocol. Both elbow branches, unfiltered."""

    def __init__(self, robot: PlanarArm | None = None):
        self.robot = robot or PlanarArm()

    def solve(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        x, y = pose[0, 3], pose[1, 3]
        d = np.sqrt(x**2 + y**2)
        l1, l2 = self.robot.l1, self.robot.l2
        if d > l1 + l2 or d < abs(l1 - l2):
            return []
        cos_q2 = np.clip((d**2 - l1**2 - l2**2) / (2 * l1 * l2), -1, 1)
        solutions = []
        for sign in (1, -1):
            q2 = sign * np.arccos(cos_q2)
            q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
            solutions.append(np.array([q1, q2]))
        return solutions


class NoCollision:
    """Every configuration is valid; ``CollisionChecker`` protocol."""

    def is_valid(self, q: np.ndarray) -> bool:
        return True


class Wall:
    """Invalid inside a slab ``lo < q[axis] < hi``, optionally only for ``|q[other]| < extent``."""

    def __init__(self, axis: int, lo: float, hi: float, extent: float | None = None):
        self.axis, self.lo, self.hi, self.extent = axis, lo, hi, extent

    def is_valid(self, q: np.ndarray) -> bool:
        inside = self.lo < q[self.axis] < self.hi
        if self.extent is not None:
            inside = inside and abs(q[1 - self.axis]) < self.extent
        return not inside
