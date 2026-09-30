# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""abort_fn is polled per iteration, per root draw, and per smoothing attempt (#109)."""

import numpy as np
from tsr import TSR

from pycbirrt import CBiRRT, CBiRRTConfig, FiniteSet, PlanningProblem, TSRConfigurationSet
from pycbirrt.testing import NoCollision, PlanarArm, PlanarIK


def _planner(**kw):
    # The reference implementation, explicitly: these tests patch its internals (the native core's cancellation
    # is covered in test_native_isolation.py).
    return CBiRRT(
        PlanarArm(), PlanarIK(), NoCollision(), CBiRRTConfig(step_size=0.1, timeout=10.0, **kw), backend="python"
    )


def _reach(planner, xy, radius=0.1):
    T = np.eye(4)
    T[:2, 3] = xy
    tsr = TSR(
        T0_w=T,
        Tw_e=np.eye(4),
        Bw=np.array([[-radius, radius], [-radius, radius], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]]),
    )
    return TSRConfigurationSet(tsr, planner.robot, planner.ik, planner.space)


class TestDuringRoots:
    def test_abort_before_the_first_goal_draw_reports_aborted_with_partial_roots(self):
        calls = []
        planner = _planner(abort_fn=lambda: calls.append(1) or True)
        problem = PlanningProblem(
            space=planner.space,
            start=FiniteSet([np.array([-0.5, 0.5])], metric=planner.space.distance),
            goal=_reach(planner, (0.0, 1.5)),  # sampled, so a draw is polled
            validator=planner.collision,
        )
        result = planner.solve(problem, seed=0)
        assert not result.success
        assert result.failure_reason.startswith("Aborted")
        assert "goal root collection" in result.failure_reason
        assert result.iterations == 0
        assert result.tree_sizes == (1, 0)  # the finite start root was gathered; no goal draw happened
        assert len(calls) == 1

    def test_finite_sets_are_not_polled_during_roots(self):
        # No draws: the first poll is the first iteration, as before #109.
        planner = _planner(abort_fn=lambda: True)
        problem = PlanningProblem(
            space=planner.space,
            start=FiniteSet([np.zeros(2)], metric=planner.space.distance),
            goal=FiniteSet([np.array([1.0, 0.5])], metric=planner.space.distance),
            validator=planner.collision,
        )
        result = planner.solve(problem, seed=0)
        assert result.failure_reason == "Aborted by user"
        assert result.tree_sizes == (1, 1)


class TestDuringSmoothing:
    def test_abort_during_smoothing_returns_the_path_found_as_success(self, monkeypatch):
        fired = {"flag": False}
        planner = _planner(abort_fn=lambda: fired["flag"], smoothing_iterations=50, smoothing_patience=50)
        real = planner._try_shortcut
        attempts = []

        def counting(problem, a, b):
            attempts.append(1)
            fired["flag"] = True  # fires on the poll before the *next* attempt
            return real(problem, a, b)

        monkeypatch.setattr(planner, "_try_shortcut", counting)
        problem = PlanningProblem(
            space=planner.space,
            start=FiniteSet([np.array([-2.5, 0.5])], metric=planner.space.distance),
            goal=FiniteSet([np.array([2.5, -0.5])], metric=planner.space.distance),
            validator=planner.collision,
        )
        result = planner.solve(problem, seed=0)
        assert result.success
        assert len(attempts) == 1  # smoothing stopped after the first attempt
        assert np.array_equal(result.path[0], [-2.5, 0.5])
        assert planner.space.distance(result.path[-1], np.array([2.5, -0.5])) < 1e-9
        assert all(planner.space.distance(a, b) <= 0.1 + 1e-9 for a, b in zip(result.path[:-1], result.path[1:]))
