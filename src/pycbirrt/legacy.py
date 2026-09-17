# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Lower the legacy ``plan(...)`` arguments into a ``PlanningProblem``.

This is the only module besides ``pycbirrt.tsr_set`` that imports from the
``tsr`` package. It reproduces the legacy composition semantics exactly:

- multiple start or goal TSRs are a union, sampled in proportion to volume;
- fixed configurations are always roots and are never sampled;
- multiple path-constraint TSRs are an intersection projected with the
  most-violated heuristic.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from tsr import TSR

from pycbirrt.config import CBiRRTConfig
from pycbirrt.interfaces import CollisionChecker, IKSolver, RobotModel
from pycbirrt.problem import PlanningProblem
from pycbirrt.sets import AllOf, AnyOf, EmptySet, FiniteSet, MostViolatedProjection, StateSet
from pycbirrt.space import JointSpace
from pycbirrt.tsr_set import TSRConfigurationSet, tsr_weights


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
    start_tsrs: Sequence[TSR] | None,
    goal_tsrs: Sequence[TSR] | None,
    constraint_tsrs: Sequence[TSR] | None,
) -> PlanningProblem:
    """Build the problem the legacy arguments describe."""

    def tsr_set(tsr: TSR) -> TSRConfigurationSet:
        return TSRConfigurationSet(
            tsr,
            robot,
            ik,
            space,
            tolerance=config.tsr_tolerance,
            max_projection_iters=config.max_projection_iters,
            progress_tolerance=config.progress_tolerance,
            max_solutions_per_pose=config.max_ik_per_pose,
        )

    def role(configs: Sequence[np.ndarray] | None, tsrs: Sequence[TSR] | None) -> StateSet:
        parts: list[StateSet] = []
        if configs:
            parts.append(FiniteSet(configs, tolerance=config.tsr_tolerance, metric=space.distance))
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
                    progress_tolerance=config.progress_tolerance,
                ),
            )

    return PlanningProblem(
        space=space,
        start=role(start, start_tsrs),
        goal=role(goal, goal_tsrs),
        validator=collision,
        path_constraint=path_constraint,
    )
