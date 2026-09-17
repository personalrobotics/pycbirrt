# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The planning problem: a state space, a start set, a goal set, and constraints."""

from __future__ import annotations

from dataclasses import dataclass

from pycbirrt.interfaces import CollisionChecker
from pycbirrt.sets import StateSet
from pycbirrt.space import JointSpace


@dataclass
class PlanningProblem:
    """Find a path from the start set to the goal set through admissible states.

    Roles (what a set is used for), independent of what a set means:

    Attributes:
        space: Joint-space geometry: limits, metric, interpolation, sampling.
        start: Where paths may begin. Must be finite (its members become tree
            roots) or sampleable (roots are drawn from it), or both.
        goal: Where paths may end. Same requirements as ``start``.
        validator: State validity, typically collision checking. Applied to
            every root and to every configuration along every edge. Not a
            set, because it supports rejection but not sampling or projection.
        path_constraint: The path-admissible set, or None. Every configuration
            on the path must be a member. If it supports projection, tree
            extensions are projected onto it; otherwise they are rejected
            when they leave it. Roots are drawn from ``start`` and ``goal``
            and rejected if they are not members.
    """

    space: JointSpace
    start: StateSet
    goal: StateSet
    validator: CollisionChecker
    path_constraint: StateSet | None = None
