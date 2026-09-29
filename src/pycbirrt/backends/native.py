# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The native backend: lower a Python ``PlanningProblem`` to ``pycbirrt._native`` and solve there.

Every component must have a native form, because the native solve releases the
GIL and calls no Python after entry. ``lower`` walks the problem and raises
``NativeUnsupported`` listing every component that blocked it; ``CBiRRT`` with
``backend="auto"`` then selects the Python backend and records the reasons.
See docs/native-design.md, "Python binding".
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from pycbirrt.config import CBiRRTConfig
from pycbirrt.exceptions import (
    AllGoalConfigurationsInCollision,
    AllGoalConfigurationsInvalid,
    AllStartConfigurationsInCollision,
    AllStartConfigurationsInvalid,
    MotionContractError,
    UnsupportedCapability,
)
from pycbirrt.legacy import legacy_index
from pycbirrt.problem import PlanningProblem
from pycbirrt.sets import AllOf, AnyOf, EmptySet, FiniteSet, MostViolatedProjection, RejectionSampling, euclidean
from pycbirrt.space import JointSpace
from pycbirrt.testing import NoCollision, Wall
from pycbirrt.tree import RRTree

try:
    from pycbirrt import _native
except ImportError:  # pragma: no cover - exercised only in builds without the extension
    _native = None


class NativeUnsupported(Exception):
    """The problem has a component with no native form. ``reasons`` lists every blocker."""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons) if self.reasons else "native backend unavailable")


def available() -> bool:
    return _native is not None


@dataclass
class Lowered:
    problem: Any  # _native.PlanningProblem
    config: Any  # _native.PlannerConfig
    space: Any  # _native.JointSpace


def _space_metric(metric, space: JointSpace) -> bool:
    """Whether ``metric`` is ``space.distance`` bound to this very space."""
    return getattr(metric, "__self__", None) is space and getattr(metric, "__func__", None) is JointSpace.distance


def _lower_space(space: JointSpace):
    angular = [] if space.angular_joints is None else [bool(a) for a in space.angular_joints]
    return _native.JointSpace(list(space.lower), list(space.upper), angular)


def _lower_set(s, where: str, space: JointSpace, nspace, reasons: list[str]):
    if isinstance(s, FiniteSet):
        configs = [list(map(float, c)) for c in s.configs]
        if s.metric is euclidean:
            return _native.FiniteSet(configs, float(s.tolerance))
        if _space_metric(s.metric, space):
            return _native.FiniteSet(configs, float(s.tolerance), nspace)
        reasons.append(f"{where}: FiniteSet metric is a Python callable; native supports euclidean or the space metric")
        return None
    if isinstance(s, EmptySet):
        return _native.EmptySet()
    if isinstance(s, AnyOf):
        children = [_lower_set(c, f"{where}[{i}]", space, nspace, reasons) for i, c in enumerate(s.children)]
        if any(c is None for c in children):
            return None
        weights = None if s.weights is None else [float(w) for w in s.weights]
        if s.metric is euclidean:
            return _native.AnyOf(children, weights)
        if _space_metric(s.metric, space):
            return _native.AnyOf(children, weights, nspace)
        reasons.append(f"{where}: AnyOf metric is a Python callable")
        return None
    if isinstance(s, AllOf):
        children = [_lower_set(c, f"{where}[{i}]", space, nspace, reasons) for i, c in enumerate(s.children)]
        if any(c is None for c in children):
            return None
        projection = sampling = None
        if s.projection is not None:
            if isinstance(s.projection, MostViolatedProjection):
                projection = _native.MostViolatedProjection(
                    int(s.projection.max_iters), float(s.projection.progress_tolerance)
                )
            else:
                reasons.append(f"{where}: projection strategy {type(s.projection).__name__} has no native form")
                return None
        if s.sampling is not None:
            if isinstance(s.sampling, RejectionSampling):
                sampling = _native.RejectionSampling(int(s.sampling.source))
            else:
                reasons.append(f"{where}: sampling strategy {type(s.sampling).__name__} has no native form")
                return None
        return _native.AllOf(children, projection, sampling)
    reasons.append(f"{where}: {type(s).__name__} has no native form in v1.5.0")
    return None


def _lower_validator(v, dof: int, reasons: list[str]):
    if isinstance(v, NoCollision):
        return _native.AcceptAll()
    if isinstance(v, Wall):
        lo, hi = [-np.inf] * dof, [np.inf] * dof
        lo[v.axis], hi[v.axis] = float(v.lo), float(v.hi)
        if v.extent is not None:
            other = 1 - v.axis
            lo[other], hi[other] = -float(v.extent), float(v.extent)
        return _native.JointBoxObstacles([(lo, hi)])
    reasons.append(f"validator: {type(v).__name__} is a Python object; native needs a sscbirrt StateValidator")
    return None


def _lower_config(config: CBiRRTConfig):
    c = _native.PlannerConfig()
    c.timeout_seconds = float(config.timeout)
    c.max_iterations = int(config.max_iterations)
    c.connection_tolerance = float(config.connection_tolerance)
    c.edge_resolution = None if config.edge_resolution is None else float(config.edge_resolution)
    c.progress_tolerance = float(config.progress_tolerance)
    c.step_size = float(config.step_size)
    c.goal_bias = float(config.goal_bias)
    c.start_bias = float(config.start_bias)
    c.extend_steps = config.extend_steps
    c.connect_steps = config.connect_steps
    c.sample_draws = int(config.tsr_samples)
    c.num_tree_roots = int(config.num_tree_roots)
    c.max_per_draw = int(config.max_ik_per_pose)
    c.smooth_path = bool(config.smooth_path)
    c.smoothing_iterations = int(config.smoothing_iterations)
    c.smoothing_patience = int(config.smoothing_patience)
    return c


def lower(problem: PlanningProblem, config: CBiRRTConfig) -> Lowered:
    """Map a Python problem and config to their native forms, or raise ``NativeUnsupported`` with every blocker."""
    if _native is None:
        raise NativeUnsupported(["pycbirrt._native is not built; install pycbirrt from a wheel with the extension"])
    reasons: list[str] = []
    nspace = _lower_space(problem.space)
    start = _lower_set(problem.start, "start", problem.space, nspace, reasons)
    goal = _lower_set(problem.goal, "goal", problem.space, nspace, reasons)
    constraint = None
    if problem.path_constraint is not None:
        constraint = _lower_set(problem.path_constraint, "path_constraint", problem.space, nspace, reasons)
    validator = _lower_validator(problem.validator, problem.space.dof, reasons)
    if problem.motion_validator is not None:
        reasons.append(f"motion_validator: {type(problem.motion_validator).__name__} is a Python object")
    if problem.sampler is not None and problem.sampler is not problem.space:
        reasons.append(f"sampler: {type(problem.sampler).__name__} is a Python object")
    if reasons:
        raise NativeUnsupported(reasons)

    np_problem = _native.PlanningProblem()
    np_problem.space = nspace
    np_problem.start = start
    np_problem.goal = goal
    np_problem.validator = validator
    np_problem.path_constraint = constraint
    return Lowered(np_problem, _lower_config(config), nspace)


def _rebuild_tree(tree) -> RRTree | None:
    if tree is None:
        return None
    nodes = tree.nodes()
    roots = [(np.array(q, dtype=float), tuple(src)) for q, parent, src in nodes if parent < 0]
    out = RRTree([q for q, _ in roots], source_indices=[s for _, s in roots])
    for q, parent, _ in nodes:
        if parent >= 0:
            out.add_node(np.array(q, dtype=float), parent)
    return out


def _raise_no_roots(e) -> None:
    report = e.report
    only = report.only_collisions()
    cls = {
        ("start", True): AllStartConfigurationsInCollision,
        ("start", False): AllStartConfigurationsInvalid,
        ("goal", True): AllGoalConfigurationsInCollision,
        ("goal", False): AllGoalConfigurationsInvalid,
    }[(e.role, only)]
    details = list(report.details)
    if report.draws:
        details.append("sampling: " + report.summary())
    raise cls(report.explicit_candidates + report.rejections(), details) from None


def solve(lowered: Lowered, seed: int | None, abort_fn: Callable[[], bool] | None):
    """Run the native solve and return a Python ``PlanResult``. Exceptions map to their Python namesakes."""
    from pycbirrt.planner import PlanResult

    token = _native.CancellationToken()
    stop = threading.Event()
    poller = None
    if abort_fn is not None:
        if abort_fn():  # the cancellation case in the artifact: aborted before any iteration
            token.cancel()
        else:

            def poll():
                while not stop.is_set():
                    if abort_fn():
                        token.cancel()
                        return
                    time.sleep(0.001)

            poller = threading.Thread(target=poll, daemon=True)
            poller.start()

    planner = _native.Planner(lowered.config)
    try:
        r = planner.solve(lowered.problem, None if seed is None else int(seed), token, True)
    except _native.NoRoots as e:
        _raise_no_roots(e)
    except _native.UnsupportedCapability as e:
        raise UnsupportedCapability(str(e)) from None
    except _native.ContractError as e:
        raise MotionContractError(str(e)) from None
    finally:
        stop.set()
        if poller is not None:
            poller.join()

    success = r.success
    start_source, goal_source = tuple(r.start_source), tuple(r.goal_source)
    return PlanResult(
        path=[np.array(q, dtype=float) for q in r.path] if success else None,
        start_index=legacy_index(start_source),
        goal_index=legacy_index(goal_source),
        iterations=r.iterations,
        planning_time=r.planning_seconds,
        tree_sizes=tuple(r.tree_sizes),
        success=success,
        failure_reason=None if success else r.reason,
        start_source=start_source,
        goal_source=goal_source,
        tree_start=_rebuild_tree(r.tree_start),
        tree_goal=_rebuild_tree(r.tree_goal),
        backend="native",
    )
