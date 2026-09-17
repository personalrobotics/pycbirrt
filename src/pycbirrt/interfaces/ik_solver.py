# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from gafro import Motor


class IKSolver(Protocol):
    """Protocol for inverse kinematics solvers.

    Implementations should provide both raw IK solving and validated solving.
    The raw `solve()` returns all kinematic solutions quickly.
    The `solve_valid()` filters for collision-free solutions within joint limits.
    """

    def solve(self, pose: "Motor | np.ndarray", q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Solve IK for a single end-effector pose (raw, unvalidated).

        Args:
            pose: Desired end-effector pose as a ``gafro.Motor``
                (a 4x4 homogeneous transform is also accepted).
            q_init: Optional initial configuration hint for iterative solvers.
                Analytical solvers may ignore this parameter.

        Returns:
            List of joint configurations (may include invalid ones)
        """
        ...

    def solve_valid(self, pose: "Motor | np.ndarray", q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Solve IK and return only valid solutions.

        Filters solutions to return only those that are:
        - Within joint limits
        - Collision-free

        Args:
            pose: Desired end-effector pose as a ``gafro.Motor``
                (a 4x4 homogeneous transform is also accepted).
            q_init: Optional initial configuration for iterative solvers

        Returns:
            List of valid joint configurations (may be empty)
        """
        ...
