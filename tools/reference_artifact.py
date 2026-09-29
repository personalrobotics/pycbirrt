# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generate or check the deterministic Python reference artifact (#94).

The artifact is the definition of "compatible with the Python reference" for
the native backends (#82, #92). It runs a fixed matrix of planning problems on
the planar reference arm with fixed seeds and records, per case:

- the problem specification as data;
- the seed and the implementation versions;
- the result: status, failure category, provenance, iterations, path;
- an independent validation report computed from the problem alone, without
  the planner: every waypoint in the joint space, the first waypoint in the
  declared start set and the last in the declared goal set, every waypoint
  admissible (validator and path constraint), every consecutive pair
  validated at ``edge_resolution`` along the space's direction, and raw joint
  continuity on angular joints.

The semantic oracle is the status, category, provenance, endpoint membership,
and validation report. Paths and iteration counts are recorded for
inspection, not compared, so the artifact is stable across platforms while
the planner's observable behavior is pinned.

Usage:
    python tools/reference_artifact.py            # regenerate tests/reference/python_reference.json
    python tools/reference_artifact.py --check    # regenerate in memory and compare semantics with the file
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tsr import TSR, TSRChain

from pycbirrt import CBiRRT, CBiRRTConfig, PlanningProblem
from pycbirrt.sets import AllOf, AnyOf, FiniteSet, MostViolatedProjection, PredicateSet, seeds
from pycbirrt.testing import NoCollision, PlanarArm, PlanarIK, Wall
from pycbirrt.tsr_set import TSRConfigurationSet, tsr_weights

ARTIFACT = Path(__file__).resolve().parent.parent / "tests" / "reference" / "python_reference.json"
SEMANTIC_KEYS = ("status", "failure_category", "start_source", "goal_source", "validation")


# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------


def _frame(x: float, y: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3], T[1, 3] = x, y
    return T


BOX = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
Y_BAND = np.array([[-2.0, 2.0], [-0.6, 0.6], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])


def _planner(config: CBiRRTConfig, validator=None) -> CBiRRT:
    robot = PlanarArm()
    return CBiRRT(robot, PlanarIK(robot), validator or NoCollision(), config)


def _finite(planner: CBiRRT, configs) -> FiniteSet:
    """Finite set under the planner's space metric, as the legacy lowering builds them (matters for angular joints)."""
    return FiniteSet(configs, tolerance=planner.config.membership_tolerance, metric=planner.space.distance)


def _tsr_set(planner: CBiRRT, tsr) -> TSRConfigurationSet:
    return TSRConfigurationSet(
        tsr, planner.robot, planner.ik, planner.space, tolerance=planner.config.membership_tolerance
    )


def cases() -> list[dict[str, Any]]:
    """Each case: name, description, seed, config, planner, problem, spec (as data)."""
    out: list[dict[str, Any]] = []

    def add(name, description, seed, config, problem, spec, planner):
        out.append(
            {
                "name": name,
                "description": description,
                "seed": seed,
                "config": config,
                "problem": problem,
                "spec": spec,
                "planner": planner,
            }
        )

    base = dict(smooth_path=True, step_size=0.2, connection_tolerance=1e-3, edge_resolution=0.05, timeout=30.0)

    # 1. fixed start to fixed goal
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    q0, q1 = np.zeros(2), np.array([1.0, 0.5])
    add(
        "fixed_to_fixed",
        "one fixed start, one fixed goal, no constraint",
        0,
        cfg,
        PlanningProblem(space=p.space, start=_finite(p, [q0]), goal=_finite(p, [q1]), validator=p.collision),
        {"start": [q0.tolist()], "goal": [q1.tolist()]},
        p,
    )

    # 2. multiple roots
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    starts = [np.array([0.0, 0.0]), np.array([0.1, 0.1])]
    goals = [np.array([1.0, 0.5]), np.array([-1.0, 0.5])]
    add(
        "multiple_roots",
        "two fixed starts, two fixed goals; provenance reports which pair connected",
        1,
        cfg,
        PlanningProblem(space=p.space, start=_finite(p, starts), goal=_finite(p, goals), validator=p.collision),
        {"start": [s.tolist() for s in starts], "goal": [g.tolist() for g in goals]},
        p,
    )

    # 3. nested finite goal with a predicate filter
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    near, far, bad = np.array([0.2, 0.1]), np.array([2.5, 2.5]), np.array([0.2, -0.5])
    goal = AnyOf([AllOf([FiniteSet([bad, near]), PredicateSet(lambda q: q[1] > 0, name="q1>0")]), FiniteSet([far])])
    add(
        "nested_finite_goal",
        "AnyOf(AllOf(FiniteSet, predicate), FiniteSet): the filtered member is expected, provenance (0, 1)",
        2,
        cfg,
        PlanningProblem(space=p.space, start=_finite(p, [np.zeros(2)]), goal=goal, validator=p.collision),
        {"goal": "AnyOf([AllOf([FiniteSet([bad, near]), q1>0]), FiniteSet([far])])", "near": near.tolist()},
        p,
    )

    # 4. wrapped joint across the seam
    cfg = CBiRRTConfig(angular_joints=(True, False), **base)
    p = _planner(cfg)
    q0, q1 = np.array([3.0, 0.0]), np.array([-3.0, 0.0])
    add(
        "wrapped_seam",
        "joint 0 angular; start and goal 0.28 rad apart across the seam; output continuous in raw values",
        3,
        cfg,
        PlanningProblem(space=p.space, start=_finite(p, [q0]), goal=_finite(p, [q1]), validator=p.collision),
        {"angular_joints": [True, False], "start": [q0.tolist()], "goal": [q1.tolist()]},
        p,
    )

    # 5. rejection-only path constraint
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    half = PredicateSet(lambda q: q[1] >= -1e-9, name="q1>=0")
    add(
        "rejection_constraint",
        "membership-only path constraint q1 >= 0, enforced by rejection",
        4,
        cfg,
        PlanningProblem(
            space=p.space,
            start=_finite(p, [np.zeros(2)]),
            goal=_finite(p, [np.array([1.0, 0.3])]),
            validator=p.collision,
            path_constraint=half,
        ),
        {"path_constraint": "PredicateSet(q1>=0)"},
        p,
    )

    # 6. projected path constraint (TSR band)
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    band = _tsr_set(p, TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=Y_BAND))
    q0, q1 = np.array([0.0, 0.6]), np.array([-0.3, 0.9])
    add(
        "projected_constraint",
        "TSR path constraint: end effector y within ±0.6, projected each step",
        5,
        cfg,
        PlanningProblem(
            space=p.space, start=_finite(p, [q0]), goal=_finite(p, [q1]), validator=p.collision, path_constraint=band
        ),
        {"path_constraint": {"tsr_Bw": Y_BAND.tolist()}},
        p,
    )

    # 7. TSR goal union
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    a = _tsr_set(p, TSR(T0_w=_frame(1.9, 0.3), Tw_e=np.eye(4), Bw=BOX))
    b = _tsr_set(p, TSR(T0_w=_frame(-1.5, 1.0), Tw_e=np.eye(4), Bw=BOX))
    add(
        "tsr_goal_union",
        "AnyOf of two TSR regions weighted by volume; provenance reports which",
        6,
        cfg,
        PlanningProblem(
            space=p.space,
            start=_finite(p, [np.zeros(2)]),
            goal=AnyOf([a, b], weights=tsr_weights([a, b])),
            validator=p.collision,
        ),
        {"goal": "AnyOf([TSR@(1.9,0.3), TSR@(-1.5,1.0)])"},
        p,
    )

    # 8. TSR chain goal
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    hinge = TSR(T0_w=_frame(1.0, 0.5), Tw_e=np.eye(4), Bw=np.array([[0, 0]] * 5 + [[-np.pi, np.pi]]))
    handle = TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=np.array([[0.3, 0.3]] + [[0, 0]] * 4 + [[-np.pi, np.pi]]))
    chain = _tsr_set(p, TSRChain(TSRs=[hinge, handle]))
    add(
        "tsr_chain_goal",
        "door chain: hinge at (1.0, 0.5) with free yaw, handle 0.3 along the door",
        7,
        cfg,
        PlanningProblem(space=p.space, start=_finite(p, [np.zeros(2)]), goal=chain, validator=p.collision),
        {"goal": "TSRChain(hinge@(1.0,0.5) yaw free, handle x=0.3 yaw free)"},
        p,
    )

    # 9. two-TSR intersection constraint with the named strategy
    cfg = CBiRRTConfig(**base)
    p = _planner(cfg)
    x_band = _tsr_set(
        p,
        TSR(
            T0_w=_frame(1.5, 0.0),
            Tw_e=np.eye(4),
            Bw=np.array([[-0.5, 0.5], [-2, 2], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]]),
        ),
    )
    y_band = _tsr_set(p, TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=Y_BAND))
    both = AllOf([x_band, y_band], projection=MostViolatedProjection())
    add(
        "allof_constraint",
        "AllOf of two TSR bands projected with MostViolatedProjection",
        8,
        cfg,
        PlanningProblem(
            space=p.space,
            start=_finite(p, [np.array([0.0, 0.6])]),
            goal=_finite(p, [np.array([-0.3, 0.9])]),
            validator=p.collision,
            path_constraint=both,
        ),
        {"path_constraint": "AllOf([x in [1,2], y in [-0.6,0.6]], MostViolatedProjection)"},
        p,
    )

    # 10. timeout: the smallest positive timeout, on a problem no single iteration can solve, so the
    # outcome does not depend on the clock's resolution (a coarse clock may let one iteration run)
    cfg = CBiRRTConfig(**{**base, "timeout": 1e-9})
    timeout_wall = Wall(axis=0, lo=0.45, hi=0.55)
    p = _planner(cfg, timeout_wall)
    add(
        "timeout",
        "a one-nanosecond timeout with a wall between start and goal: the deadline reports a timeout",
        9,
        cfg,
        PlanningProblem(
            space=p.space,
            start=_finite(p, [np.zeros(2)]),
            goal=_finite(p, [np.array([1.0, 0.0])]),
            validator=timeout_wall,
        ),
        {"timeout": 1e-9, "validator": "Wall(q0 in (0.45, 0.55))"},
        p,
    )

    # 11. cancellation
    cfg = CBiRRTConfig(**{**base, "abort_fn": lambda: True})
    p = _planner(cfg)
    add(
        "cancellation",
        "abort_fn always true: the first iteration reports an abort",
        10,
        cfg,
        PlanningProblem(
            space=p.space,
            start=_finite(p, [np.zeros(2)]),
            goal=_finite(p, [np.array([1.0, 0.5])]),
            validator=p.collision,
        ),
        {"abort": True},
        p,
    )

    # 12. unreachable: a wall across all of joint 1 between start and goal
    cfg = CBiRRTConfig(**{**base, "max_iterations": 300, "timeout": 30.0})
    wall = Wall(axis=0, lo=0.45, hi=0.55)
    p = _planner(cfg, wall)
    add(
        "unreachable",
        "a wall at q0 in (0.45, 0.55) across all q1 separates start and goal; max iterations reached",
        11,
        cfg,
        PlanningProblem(
            space=p.space, start=_finite(p, [np.zeros(2)]), goal=_finite(p, [np.array([1.0, 0.0])]), validator=wall
        ),
        {"validator": "Wall(q0 in (0.45, 0.55))", "max_iterations": 300},
        p,
    )

    return out


# ---------------------------------------------------------------------------
# Independent validation
# ---------------------------------------------------------------------------


def validate(problem: PlanningProblem, config: CBiRRTConfig, path: list[np.ndarray]) -> dict[str, Any]:
    """Check a path against the problem using only the problem's own operations."""
    space = problem.space

    def admissible(q):
        if not space.contains(q):
            return False
        if not problem.validator.is_valid(q):
            return False
        return problem.path_constraint is None or problem.path_constraint.contains(q)

    resolution = config.edge_resolution or config.step_size
    edges_ok = True
    max_raw_step = 0.0
    for a, b in zip(path[:-1], path[1:]):
        d = space.direction(a, b)
        n = max(1, int(np.ceil(np.linalg.norm(d) / resolution)))
        for i in range(1, n + 1):
            if not admissible(a + (i / n) * d):
                edges_ok = False
        max_raw_step = max(max_raw_step, float(np.abs(np.asarray(b) - np.asarray(a)).max()))
    return {
        "all_in_space": all(space.contains(q) for q in path),
        "first_in_start_set": bool(problem.start.contains(path[0])),
        "last_in_goal_set": bool(problem.goal.contains(path[-1])),
        "all_admissible": all(admissible(q) for q in path),
        "edges_validated_at_resolution": edges_ok,
        "raw_steps_within_step_size": max_raw_step <= config.step_size + 1e-9,
        "max_raw_step": max_raw_step,
        "waypoints": len(path),
    }


def failure_category(reason: str | None) -> str | None:
    if reason is None:
        return None
    for prefix, category in (("Timeout", "timeout"), ("Aborted", "aborted"), ("Max iterations", "max_iterations")):
        if reason.startswith(prefix):
            return category
    return "other"


# ---------------------------------------------------------------------------
# Generation and comparison
# ---------------------------------------------------------------------------


def versions() -> dict[str, str]:
    def v(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    return {
        "pycbirrt": v("pycbirrt"),
        "sstsr": v("sstsr"),
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def run_case(case: dict[str, Any], backend: str = "python") -> dict[str, Any]:
    """Run one case. With backend="native", a case the native core cannot lower is recorded as unsupported."""
    planner: CBiRRT = case["planner"]
    if backend == "native":
        from pycbirrt.backends import native

        try:
            lowered = native.lower(case["problem"], case["config"])
        except native.NativeUnsupported as e:
            return {
                "name": case["name"],
                "description": case["description"],
                "seed": case["seed"],
                "spec": case["spec"],
                "status": "unsupported",
                "failure_category": None,
                "start_source": [],
                "goal_source": [],
                "iterations": 0,
                "path": None,
                "validation": None,
                "unsupported_reasons": list(e.reasons),
            }
        result = native.solve(lowered, case["seed"], case["config"].abort_fn)
    else:
        result = planner.solve(case["problem"], seed=case["seed"])
    record = {
        "name": case["name"],
        "description": case["description"],
        "seed": case["seed"],
        "spec": case["spec"],
        "status": "success" if result.success else "failure",
        "failure_category": failure_category(result.failure_reason),
        "start_source": list(result.start_source),
        "goal_source": list(result.goal_source),
        "iterations": result.iterations,
        "path": [q.tolist() for q in result.path] if result.success else None,
        "validation": validate(case["problem"], case["config"], result.path) if result.success else None,
    }
    return record


def generate(backend: str = "python") -> dict[str, Any]:
    return {
        "artifact": f"pycbirrt {backend} behavior on the reference matrix",
        "issue": "https://github.com/personalrobotics/pycbirrt/issues/94",
        "backend": backend,
        "versions": versions(),
        "cases": [run_case(c, backend) for c in cases()],
    }


def uniquely_rooted(case_name: str) -> bool:
    """Whether both roles of the named case have exactly one admissible root, so provenance is comparable."""
    for c in cases():
        if c["name"] == case_name:
            return len(seeds(c["problem"].start)) == 1 and len(seeds(c["problem"].goal)) == 1
    raise KeyError(case_name)


def parity_mismatches(python_artifact: dict[str, Any], native_artifact: dict[str, Any]) -> list[str]:
    """Where a native run disagrees with the Python artifact on the semantic view, for the cases native supports.

    Provenance is compared only where the reached root is unique (docs/native-design.md, parity note).
    """
    by_name = {c["name"]: c for c in semantic_view(python_artifact)}
    out = []
    for nat in semantic_view(native_artifact):
        if nat["status"] == "unsupported":
            continue
        py = by_name[nat["name"]]
        keys = ["status", "failure_category", "validation"]
        if uniquely_rooted(nat["name"]):
            keys += ["start_source", "goal_source"]
        for k in keys:
            if nat[k] != py[k]:
                out.append(f"{nat['name']}.{k}: python={py[k]!r} native={nat[k]!r}")
    return out


def semantic_view(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """The comparable part: no paths, no iteration counts, no versions."""
    view = []
    for case in artifact["cases"]:
        entry = {k: case[k] for k in ("name",) + SEMANTIC_KEYS}
        if entry["validation"] is not None:
            entry["validation"] = {
                k: v for k, v in entry["validation"].items() if k not in ("max_raw_step", "waypoints")
            }
        view.append(entry)
    return view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare a fresh run's semantics with the stored artifact")
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    parser.add_argument(
        "--backend",
        choices=["python", "native"],
        default="python",
        help="which implementation to run; with --check, native is compared with the stored Python artifact",
    )
    args = parser.parse_args(argv)

    fresh = generate(args.backend)
    if args.check and args.backend == "native":
        stored = json.loads(ARTIFACT.read_text())
        mismatches = parity_mismatches(stored, fresh)
        supported = [c["name"] for c in fresh["cases"] if c["status"] != "unsupported"]
        if mismatches:
            print("PARITY MISMATCH between the native core and the Python artifact:", file=sys.stderr)
            for m in mismatches:
                print(f"  {m}", file=sys.stderr)
            return 1
        print(f"native matches the Python artifact on {len(supported)} supported cases: {', '.join(supported)}")
        return 0
    if args.check:
        stored = json.loads(args.output.read_text())
        if semantic_view(stored) != semantic_view(fresh):
            print("MISMATCH: the Python planner's semantics on the reference matrix changed", file=sys.stderr)
            for old, new in zip(semantic_view(stored), semantic_view(fresh)):
                if old != new:
                    print(f"  {old['name']}:\n    stored: {old}\n    fresh:  {new}", file=sys.stderr)
            return 1
        exact = json.dumps(stored, sort_keys=True) == json.dumps(fresh, sort_keys=True)
        print(f"semantics match ({len(fresh['cases'])} cases); bit-for-bit: {exact} (versions {fresh['versions']})")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fresh, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.output} ({len(fresh['cases'])} cases, versions {fresh['versions']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
