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

