# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Isolation, cancellation latency, provenance, and cost breakdown on the native backend (#89, #139)."""

import threading
import time

import numpy as np
import pytest

from sscbirrt import CBiRRT, CBiRRTConfig, FiniteSet, PlanningProblem
from sscbirrt.backends import native
from sscbirrt.testing import NoCollision, PlanarArm, PlanarIK, Wall

pytest.importorskip("sscbirrt._native")


def _planner(backend="native", collision=None, **kw):
    cfg = CBiRRTConfig(step_size=0.2, edge_resolution=0.05, timeout=30.0, **kw)
    return CBiRRT(PlanarArm(), PlanarIK(), collision or NoCollision(), cfg, backend=backend)


def _finite(planner, qs):
    return FiniteSet(qs, tolerance=1e-3, metric=planner.space.distance)


class TestProvenanceAndStats:
    def test_both_backends_report_versions_and_counts(self):
        for backend in ("python", "native"):
            planner = _planner(backend)
            problem = PlanningProblem(
                space=planner.space,
                start=_finite(planner, [np.zeros(2)]),
                goal=_finite(planner, [np.array([1.0, 0.5])]),
                validator=planner.collision,
            )
            r = planner.solve(problem, seed=0)
            assert r.success and r.provenance["backend"] == backend
            assert "sscbirrt" in r.provenance and "sstsr" in r.provenance
            assert r.stats["state_checks"] > 0 and r.stats["edge_checks"] > 0
        native_result = _planner("native").solve(problem, seed=0)
        assert native_result.stats["seconds_search"] >= 0.0 and "seconds_edge_checks" in native_result.stats


@pytest.mark.skipif(
    not native.available() or not __import__("sscbirrt.backends.native_mujoco", fromlist=["x"]).available(),
    reason="native MuJoCo scene",
)
class TestMuJoCoIsolation:
    XML = """
    <mujoco><compiler angle="radian"/><worldbody>
      <geom name="floor" type="plane" size="3 3 0.1"/>
      <geom name="wall" type="box" pos="0.47 0 0.3" size="0.05 0.5 0.3"/>
      <body name="arm/base" pos="0 0 0.1"><joint name="j0" type="hinge" axis="0 0 1" limited="true" range="-3 3"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
        <body name="arm/link1" pos="0 0 0.1"><joint name="j1" type="hinge" axis="0 1 0" limited="true" range="-2 2"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0.4 0 0"/>
          <body name="arm/gripper/base" pos="0.4 0 0"><geom type="box" size="0.02 0.03 0.02"/></body>
        </body>
      </body>
    </worldbody></mujoco>"""

    def _problem(self, backend):
        import mujoco

        from sscbirrt.backends import native_mujoco as nm

        model = mujoco.MjModel.from_xml_string(self.XML)
        data = mujoco.MjData(model)
        scene = nm.NativeScene.from_model(model, ["j0", "j1"])
        checker = nm.NativeCollisionChecker(scene, nm.Snapshot.capture(scene, data))
        planner = CBiRRT(
            PlanarArm(),
            PlanarIK(),
            checker,
            CBiRRTConfig(step_size=0.1, edge_resolution=0.02, timeout=30.0),
            backend=backend,
        )
        # planar arm limits ±pi cover the scene's ±3, ±2 ranges only partly; use a space from the scene limits
        from sscbirrt import JointSpace

        space = JointSpace(*scene.joint_limits)
        start = FiniteSet([np.array([0.0, -1.2])], tolerance=1e-3, metric=space.distance)
        goal = FiniteSet([np.array([2.5, -1.2])], tolerance=1e-3, metric=space.distance)
        return planner, PlanningProblem(space=space, start=start, goal=goal, validator=checker), checker

    def test_provenance_carries_scene_and_snapshot_identity(self):
        planner, problem, checker = self._problem("native")
        r = planner.solve(problem, seed=0)
        assert r.success and r.backend == "native"
        assert r.provenance["scene_mjb_sha256"] == checker.scene.provenance["mjb_sha256"]
        assert r.provenance["snapshot_sha256"] == checker.snapshot.sha256
        assert r.provenance["mujoco"] == "3.14.0"

    def test_concurrent_solves_equal_their_single_threaded_runs(self):
        planner, problem, _ = self._problem("native")
        expected = {s: planner.solve(problem, seed=s).path for s in range(4)}
        results = {}
        errors = []

        def worker(seed):
            try:
                results[seed] = planner.solve(problem, seed=seed).path
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        for s in range(4):
            assert len(results[s]) == len(expected[s]) and all(
                np.array_equal(a, b) for a, b in zip(results[s], expected[s])
            )


def test_cancellation_returns_within_the_documented_bound():
    """A cancel request is honored within one poll plus one edge's remaining state checks (#89)."""
    planner = _planner("native", collision=Wall(axis=0, lo=0.45, hi=0.55), max_iterations=10**7)
    problem = PlanningProblem(
        space=planner.space,
        start=_finite(planner, [np.zeros(2)]),
        goal=_finite(planner, [np.array([1.0, 0.0])]),  # across the wall: the search would run to max_iterations
        validator=planner.collision,
    )
    deadline = time.perf_counter() + 0.05
    planner.config.abort_fn = lambda: time.perf_counter() > deadline
    t0 = time.perf_counter()
    r = planner.solve(problem, seed=0)
    elapsed = time.perf_counter() - t0
    assert not r.success and r.failure_reason.startswith("Aborted"), r.failure_reason
    # The wrapper polls abort_fn every millisecond and the core checks the token once per iteration, so the
    # bound is the poll interval plus one iteration's work; 0.5 s leaves two orders of magnitude of slack.
    assert 0.05 <= elapsed < 0.5, elapsed
