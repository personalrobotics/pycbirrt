# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Lower the legacy ``plan(...)`` arguments into a ``PlanningProblem``.

This is the only module besides ``sscbirrt.tsr_set`` that imports from the
``tsr`` package. It reproduces the legacy composition semantics exactly:

- multiple start or goal TSRs are a union, sampled in proportion to volume;
- fixed configurations are always roots and are never sampled;
- multiple path-constraint TSRs are an intersection projected with the
  most-violated heuristic.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from tsr import TSR, TSRChain

from sscbirrt.config import CBiRRTConfig
from sscbirrt.interfaces import CollisionChecker, IKSolver, RobotModel
from sscbirrt.problem import PlanningProblem
from sscbirrt.sets import AllOf, AnyOf, EmptySet, FiniteSet, MostViolatedProjection, StateSet
from sscbirrt.space import JointSpace
from sscbirrt.tsr_set import TSRConfigurationSet, tsr_weights


def legacy_index(source: tuple[int, ...]) -> int:
    """Map a root's provenance to the legacy ``start_index`` / ``goal_index``.

    The lowering below arranges that the last element of the source path is
    the index into the config list or the TSR list, whichever produced the
    root. A root with no provenance (single implicit set) maps to 0.
    """
    return source[-1] if source else 0


def legacy_problem(
    robot: RobotModel,
    ik: IKSolver,
    collision: CollisionChecker,
    space: JointSpace,
    config: CBiRRTConfig,
    start: Sequence[np.ndarray] | None,
    goal: Sequence[np.ndarray] | None,
    start_tsrs: Sequence[TSR | TSRChain] | None,
    goal_tsrs: Sequence[TSR | TSRChain] | None,
    constraint_tsrs: Sequence[TSR | TSRChain] | None,
) -> PlanningProblem:
    """Build the problem the legacy arguments describe.

    Each entry of a TSR list may be a ``TSR`` or a ``TSRChain``; a chain is one
    region (one alternative in a union, one factor in an intersection).
    """

    def tsr_set(tsr: TSR | TSRChain) -> TSRConfigurationSet:
        return TSRConfigurationSet(
            tsr,
            robot,
            ik,
            space,
            tolerance=config.membership_tolerance,
            max_projection_iters=config.max_projection_iters,
            progress_tolerance=config.projection_progress_tolerance,
        )

    def role(configs: Sequence[np.ndarray] | None, tsrs: Sequence[TSR | TSRChain] | None) -> StateSet:
        parts: list[StateSet] = []
        if configs:
            parts.append(FiniteSet(configs, tolerance=config.membership_tolerance, metric=space.distance))
        if tsrs:
            sets = [tsr_set(t) for t in tsrs]
            parts.append(AnyOf(sets, weights=tsr_weights(sets)))
        if not parts:
            # Nothing provided: the planner reports it when it collects roots,
            # after validating the other role, as the legacy code did.
            return EmptySet()
        if len(parts) == 1:
            return parts[0]
        # Fixed configs are enumerated as roots, never sampled: weight 0.
        return AnyOf(parts, weights=[0.0, 1.0])

    path_constraint: StateSet | None = None
    if constraint_tsrs:
        sets = [tsr_set(t) for t in constraint_tsrs]
        if len(sets) == 1:
            path_constraint = sets[0]
        else:
            path_constraint = AllOf(
                sets,
                projection=MostViolatedProjection(
                    max_iters=config.max_projection_iters,
                    progress_tolerance=config.projection_progress_tolerance,
                ),
            )

    return PlanningProblem(
        space=space,
        start=role(start, start_tsrs),
        goal=role(goal, goal_tsrs),
        validator=collision,
        path_constraint=path_constraint,
    )
