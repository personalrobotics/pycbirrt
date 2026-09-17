# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Local-motion validation: the boundary every tree edge passes through.

A state validator answers whether one configuration is valid. A motion
validator answers whether the local motion between two configurations is
valid, and returns the configurations to store along it. The planner uses
one motion validator for ordinary growth, the final connection between
trees, and shortcut smoothing. The default discretizes the straight segment
at ``edge_resolution`` and checks each sample; a backend may supply
continuous collision checking or any stricter method through
``PlanningProblem.motion_validator``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from pycbirrt.space import JointSpace


@dataclass
class LocalMotion:
    """The validated part of a local motion from ``q_from`` toward ``q_to``.

    Attributes:
        configs: Validated configurations after ``q_from``, in order, to be
            stored as consecutive tree nodes. Empty if the first sample was
            invalid. If ``reached`` is True the last entry is ``q_to`` exactly
            and, for a nonzero motion, ``configs`` is nonempty; the planner
            raises ``MotionContractError`` otherwise.
        reached: Whether the whole motion was valid and ends at ``q_to``.
    """

    configs: list[np.ndarray] = field(default_factory=list)
    reached: bool = False


@runtime_checkable
class MotionValidator(Protocol):
    """Validate the local motion between two configurations.

    Returns the validated prefix of the motion (see ``LocalMotion``); a
    partially valid motion returns the configurations up to the first invalid
    one with ``reached=False``, which lets tree growth keep its progress.
    """

    def validate(self, q_from: np.ndarray, q_to: np.ndarray) -> LocalMotion: ...


class DiscreteMotionValidator:
    """Straight-line motion, checked every ``resolution`` along the segment.

    Samples ``q_from + (i/n) * direction`` for ``i = 1..n`` with
    ``n = ceil(distance / resolution)`` under the space's wrapped direction;
    the last sample is ``q_to`` exactly. Each sample must satisfy
    ``is_admissible``. Validity is therefore discrete: an obstacle thinner
    than ``resolution`` can be stepped over. Choose the resolution against
    the thinnest feature you must not miss.
    """

    def __init__(self, space: JointSpace, is_admissible: Callable[[np.ndarray], bool], resolution: float):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.space = space
        self.is_admissible = is_admissible
        self.resolution = resolution

    def validate(self, q_from: np.ndarray, q_to: np.ndarray) -> LocalMotion:
        space = self.space
        if not space.contains(q_to):
            return LocalMotion()
        q_from = np.asarray(q_from, dtype=float)
        direction = space.direction(q_from, q_to)
        distance = float(np.linalg.norm(direction))
        if distance == 0.0:
            return LocalMotion(configs=[], reached=True)
        n_steps = max(1, int(np.ceil(distance / self.resolution)))
        configs: list[np.ndarray] = []
        for i in range(1, n_steps + 1):
            q = np.array(q_to, dtype=float) if i == n_steps else q_from + (i / n_steps) * direction
            if not self.is_admissible(q):
                return LocalMotion(configs=configs, reached=False)
            configs.append(q)
        return LocalMotion(configs=configs, reached=True)
