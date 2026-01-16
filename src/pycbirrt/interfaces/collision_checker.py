from typing import Protocol
import numpy as np


class CollisionChecker(Protocol):
    """Protocol for collision checking."""

    def is_valid(self, q: np.ndarray) -> bool:
        """Check if a configuration is collision-free.

        Args:
            q: Joint configuration array

        Returns:
            True if collision-free, False otherwise
        """
        ...

    def is_valid_batch(self, qs: np.ndarray) -> np.ndarray:
        """Check multiple configurations for collisions.

        Args:
            qs: Array of shape (N, dof) of joint configurations

        Returns:
            Boolean array of shape (N,) indicating validity
        """
        ...
