# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The native backend: lowering, explicit fallback, no Python callbacks, and semantic parity (#118)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from tsr import TSR

from pycbirrt import (
    AllStartConfigurationsInCollision,
    CBiRRT,
    CBiRRTConfig,
    FiniteSet,
    PlanningProblem,
    PredicateSet,
    TSRConfigurationSet,
)
from pycbirrt.backends import native
from pycbirrt.sets import seeds
from pycbirrt.testing import NoCollision, PlanarArm, PlanarIK, Wall

pytest.importorskip("pycbirrt._native")

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def artifact_tool():
    spec = importlib.util.spec_from_file_location("reference_artifact", ROOT / "tools" / "reference_artifact.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["reference_artifact"] = module
    spec.loader.exec_module(module)
    return module


def _planner(backend, collision=None, **kw):
    cfg = CBiRRTConfig(step_size=0.2, edge_resolution=0.05, timeout=30.0, **kw)
    return CBiRRT(PlanarArm(), PlanarIK(), collision or NoCollision(), cfg, backend=backend)


def _finite(planner, configs):
    return FiniteSet(configs, tolerance=planner.config.membership_tolerance, metric=planner.space.distance)


class TestLowering:
    def test_finite_problem_lowers_and_solves_natively(self):
        planner = _planner("native")
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.5])]),
            validator=planner.collision,
        )
        result = planner.solve(problem, seed=0)
        assert result.success and result.backend == "native" and result.backend_reasons == ()
        assert np.array_equal(result.path[0], np.zeros(2))
        assert planner.space.distance(result.path[-1], np.array([1.0, 0.5])) <= 1e-3
        assert result.start_source == (0,) and result.goal_source == (0,) and result.goal_index == 0
        assert result.tree_start is not None and len(result.tree_start) == result.tree_sizes[0]

    def test_wall_lowers_to_a_box(self):
        planner = _planner("native", collision=Wall(axis=0, lo=0.45, hi=0.55), max_iterations=300)
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.0])]),
            validator=planner.collision,
        )
        result = planner.solve(problem, seed=0)
        assert not result.success and result.failure_reason.startswith("Max iterations")
        assert all(n.config[0] <= 0.45 for n in result.tree_start.nodes)

    def test_unsupported_components_are_all_reported(self):
        planner = _planner("native")
        tsr = TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([[-0.1, 0.1], [-0.1, 0.1], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]]),
        )
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=TSRConfigurationSet(tsr, planner.robot, planner.ik, planner.space),
            validator=planner.collision,
            path_constraint=PredicateSet(lambda q: True, name="p"),
        )
        with pytest.raises(native.NativeUnsupported) as info:
            planner.solve(problem, seed=0)
        reasons = info.value.reasons
        assert any(r.startswith("goal: TSRConfigurationSet") for r in reasons)
        assert any(r.startswith("path_constraint: PredicateSet") for r in reasons)
        assert len(reasons) == 2

    def test_auto_selects_python_and_records_why(self):
        planner = _planner("auto")
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.5])]),
            validator=planner.collision,
            path_constraint=PredicateSet(lambda q: q[1] >= -1e-9, name="q1>=0"),
        )
        result = planner.solve(problem, seed=4)
        assert result.success and result.backend == "python"
        assert result.backend_reasons == ("path_constraint: PredicateSet has no native form in v1.5.0",)

    def test_default_backend_is_python_and_plan_is_unchanged(self):
        planner = _planner("python")
        assert planner.backend == "python"
        result = planner.plan(start=np.zeros(2), goal=np.array([1.0, 0.5]), seed=0, return_details=True)
        assert result.success and result.backend == "python" and result.backend_reasons == ()
        with pytest.raises(ValueError, match="backend must be"):
            _planner("cuda")

    def test_no_roots_maps_to_the_python_exception(self):
        planner = _planner("native", collision=Wall(axis=0, lo=-1.0, hi=1.0))
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([2.0, 2.0])]),
            validator=planner.collision,
        )
        with pytest.raises(AllStartConfigurationsInCollision):
            planner.solve(problem, seed=0)

    def test_abort_fn_is_honored(self):
        planner = _planner("native", abort_fn=lambda: True)
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.5])]),
            validator=planner.collision,
        )
        result = planner.solve(problem, seed=0)
        assert not result.success and result.failure_reason.startswith("Aborted")


class TestNoCallbacks:
    def test_native_solve_makes_no_python_calls(self):
        planner = _planner("native")
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.5])]),
            validator=planner.collision,
        )
        lowered = native.lower(problem, planner.config)
        nplanner = native._native.Planner(lowered.config)
        calls = []

        def profiler(frame, event, arg):
            if event == "call":
                calls.append(frame.f_code.co_name)

        sys.setprofile(profiler)
        try:
            r = nplanner.solve(lowered.problem, 0, None, True)
        finally:
            sys.setprofile(None)
        assert r.success
        assert calls == []  # nothing in Python ran between entry and exit


class TestParityWithTheArtifact:
    """Native and Python agree on the semantic view for every artifact case the native core supports."""

    def test_semantic_parity(self, artifact_tool):
        lowered_any = False
        for case in artifact_tool.cases():
            cfg, problem, planner = case["config"], case["problem"], case["planner"]
            try:
                lowered = native.lower(problem, cfg)
            except native.NativeUnsupported:
                continue
            lowered_any = True
            py = planner.solve(problem, seed=case["seed"])
            nat = native.solve(lowered, case["seed"], cfg.abort_fn)
            assert nat.success == py.success, case["name"]
            assert artifact_tool.failure_category(nat.failure_reason) == artifact_tool.failure_category(
                py.failure_reason
            ), case["name"]
            if py.success:
                v = artifact_tool.validate(problem, cfg, nat.path)
                assert all(
                    v[k]
                    for k in (
                        "all_in_space",
                        "first_in_start_set",
                        "last_in_goal_set",
                        "all_admissible",
                        "edges_validated_at_resolution",
                        "raw_steps_within_step_size",
                    )
                ), (case["name"], v)
                # Provenance is compared only where the reached root is unique.
                if len(seeds(problem.start)) == 1 and len(seeds(problem.goal)) == 1:
                    assert (nat.start_source, nat.goal_source) == (py.start_source, py.goal_source), case["name"]
        assert lowered_any
