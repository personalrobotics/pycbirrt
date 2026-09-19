# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

from typing import Protocol

import numpy as np


class IKSolver(Protocol):
    """Protocol for inverse kinematics solvers: one capability, ``solve``.

    The planner never filters inside the solver. Joint limits are enforced by
    ``JointSpace`` and collision by the problem's validator, so a solver should
    return every kinematic solution it knows of, including, for a joint whose
    range exceeds one turn, every in-limit winding of each geometric branch.
    Discarding branches here would hide admissible configurations from the
    planner (see #36 and #63).
    """

    def solve(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Solve IK for a single end-effector pose.

        Args:
            pose: 4x4 homogeneous transform of the desired end-effector pose,
                in the same base frame and end-effector frame the robot model
                uses for forward kinematics.
            q_init: Optional configuration hint. Iterative solvers start from
                it; enumerative solvers may use it as a seed to order and
                rewrap solutions. May be ignored.

        Returns:
            All joint configurations found, each a 1-D array of length dof,
            possibly empty. No collision or limit filtering is implied.
        """
        ...
