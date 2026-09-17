# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Reached means connected by a validated edge; shortcuts are validated and shorter (#47).

The invariant: every consecutive pair of waypoints on any returned path has been
checked at ``edge_resolution`` by the same routine used for ordinary growth.
"""

import numpy as np

from pycbirrt import CBiRRT, CBiRRTConfig, PlanningProblem
from pycbirrt.sets import FiniteSet
from pycbirrt.tree import RRTree
from tests.test_planner import MockCollisionChecker, MockIKSolver, MockRobotModel


class Slab:
    """Invalid inside a thin wall 0.02 < q[0] < 0.03 for |q[1]| < 0.5; passable around it."""

    def is_valid(self, q):
        return not (0.02 < q[0] < 0.03 and abs(q[1]) < 0.5)


class Disk:
    """Invalid inside a disk in configuration space."""

    def __init__(self, center, radius):
        self.center, self.radius = np.asarray(center, dtype=float), radius

    def is_valid(self, q):
        return np.linalg.norm(np.asarray(q) - self.center) > self.radius


def make_planner(validator=None, **cfg):
    robot = MockRobotModel()
    collision = validator or MockCollisionChecker()
    return CBiRRT(robot, MockIKSolver(robot, MockCollisionChecker()), collision, CBiRRTConfig(**cfg))


def problem(planner, start, goal):
    return PlanningProblem(
        space=planner.space, start=FiniteSet([start]), goal=FiniteSet([goal]), validator=planner.collision
    )


def segment_samples(space, a, b, resolution):
    """The samples the edge routine checks between a and b, endpoint included."""
    d = space.direction(a, b)
    n = max(1, int(np.ceil(np.linalg.norm(d) / resolution)))
    return [a + (i / n) * d for i in range(1, n + 1)]


def assert_path_validated(planner, path):
    res = planner.config.edge_resolution or planner.config.step_size
    for a, b in zip(path[:-1], path[1:]):
        for q in segment_samples(planner.space, a, b, res):
            assert planner.collision.is_valid(q), (a, b, q)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


class TestExactConnection:
    def test_thin_slab_within_tolerance_is_not_reached(self):
        """The issue's reproduction: both endpoints valid, invalid slab between, within tolerance."""
        planner = make_planner(Slab(), connection_tolerance=0.1, edge_resolution=0.005)
        q0, q1 = np.zeros(2), np.array([0.05, 0.0])
        tree = RRTree(q0)
        idx, reached = planner._grow(problem(planner, q0, q1), tree, q1)
        assert not reached
        assert all(planner.collision.is_valid(n.config) for n in tree.nodes)
        assert not any(np.array_equal(n.config, q1) for n in tree.nodes)

    def test_within_tolerance_valid_segment_adds_exact_target(self):
        planner = make_planner(connection_tolerance=0.1, edge_resolution=0.01)
        q0, q1 = np.zeros(2), np.array([0.05, 0.0])
        tree = RRTree(q0)
        idx, reached = planner._grow(problem(planner, q0, q1), tree, q1)
        assert reached
        assert np.array_equal(tree.nodes[idx].config, q1)
        # Five interior samples at 0.01 plus the endpoint
        assert len(tree) == 1 + 5

    def test_above_tolerance_grows_then_connects_exactly(self):
        planner = make_planner(connection_tolerance=1e-3, step_size=0.1)
        q0, q1 = np.zeros(2), np.array([0.37, 0.0])
        tree = RRTree(q0)
        idx, reached = planner._grow(problem(planner, q0, q1), tree, q1)
        assert reached
        assert np.array_equal(tree.nodes[idx].config, q1)
        path = tree.get_path_to_root(idx)
        assert np.array_equal(path[0], q0) and np.array_equal(path[-1], q1)
        assert max(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])) <= 0.1 + 1e-9

    def test_exact_target_already_in_tree_is_reached_without_duplicate(self):
        planner = make_planner()
        q0 = np.zeros(2)
        tree = RRTree(q0)
        idx, reached = planner._grow(problem(planner, q0, q0), tree, q0)
        assert reached and idx == 0 and len(tree) == 1

    def test_out_of_space_target_is_never_reached(self):
        planner = make_planner(connection_tolerance=100.0)
        tree = RRTree(np.zeros(2))
        idx, reached = planner._grow(problem(planner, np.zeros(2), np.zeros(2)), tree, np.array([10.0, 0.0]))
        assert not reached and len(tree) == 1

    def test_solve_never_crosses_the_slab(self):
        """End to end: with a generous connection tolerance, the final join is still validated."""
        planner = make_planner(Slab(), connection_tolerance=0.1, edge_resolution=0.005, smooth_path=False)
        q0, q1 = np.zeros(2), np.array([0.05, 0.0])
        for seed in range(3):
            result = planner.solve(problem(planner, q0, q1), seed=seed)
            assert result.success
            assert np.array_equal(result.path[0], q0) and np.array_equal(result.path[-1], q1)
            assert_path_validated(planner, result.path)

    def test_join_has_no_duplicate_waypoint(self):
        planner = make_planner(smooth_path=False)
        result = planner.solve(problem(planner, np.zeros(2), np.array([1.0, 0.5])), seed=0)
        assert result.success
        for a, b in zip(result.path[:-1], result.path[1:]):
            assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Shortcuts and smoothing
# ---------------------------------------------------------------------------


class TestShortcuts:
    def test_immediate_success_keeps_start(self):
        planner = make_planner(connection_tolerance=0.1)
        q0, q1 = np.zeros(2), np.array([0.05, 0.0])
        shortcut = planner._try_shortcut(problem(planner, q0, q1), q0, q1)
        assert shortcut is not None
        assert np.array_equal(shortcut[0], q0) and np.array_equal(shortcut[-1], q1)

    def test_identical_endpoints(self):
        planner = make_planner()
        q0 = np.array([0.3, 0.2])
        shortcut = planner._try_shortcut(problem(planner, q0, q0), q0, q0)
        assert shortcut is not None and len(shortcut) == 1 and np.array_equal(shortcut[0], q0)

    def test_shortcut_through_slab_fails(self):
        planner = make_planner(Slab(), connection_tolerance=0.1, edge_resolution=0.005)
        q0, q1 = np.zeros(2), np.array([0.05, 0.0])
        assert planner._try_shortcut(problem(planner, q0, q1), q0, q1) is None

    def test_shortcut_ends_exactly_at_target(self):
        planner = make_planner(step_size=0.1)
        q0, q1 = np.zeros(2), np.array([0.33, 0.21])
        shortcut = planner._try_shortcut(problem(planner, q0, q1), q0, q1)
        assert np.array_equal(shortcut[-1], q1)


class TestSmoothing:
    def zigzag(self):
        return [np.array([0.1 * k, 0.2 * (k % 2)]) for k in range(8)]

    def test_preserves_endpoints_exactly(self):
        planner = make_planner(step_size=0.3, smoothing_iterations=100)
        for seed in range(5):
            planner._rng = np.random.default_rng(seed)
            path = self.zigzag()
            smoothed = planner._smooth_path(problem(planner, path[0], path[-1]), path)
            assert np.array_equal(smoothed[0], path[0])
            assert np.array_equal(smoothed[-1], path[-1])

    def test_every_segment_validated_after_smoothing(self):
        # A small disk clear of every zigzag segment (nearest is 0.045 away) but in the way of
        # some straight shortcuts, e.g. (0,0) -> (0.3,0.2) passes within 0.027 of its center.
        disk = Disk([0.1, 0.1], 0.03)
        planner = make_planner(disk, step_size=0.3, edge_resolution=0.01, smoothing_iterations=200)
        path = self.zigzag()
        assert_path_validated(planner, path)  # the input path is valid segment by segment
        for seed in range(5):
            planner._rng = np.random.default_rng(seed)
            smoothed = planner._smooth_path(problem(planner, path[0], path[-1]), list(path))
            assert_path_validated(planner, smoothed)

    def test_accepts_only_shorter_shortcuts(self, monkeypatch):
        planner = make_planner()
        path = [np.array([0.0, 0.0]), np.array([0.1, 0.1]), np.array([0.2, 0.0])]  # a bend, length 0.283
        prob = problem(planner, path[0], path[-1])

        # A "shortcut" with the same waypoint count but longer must be rejected
        detour = [path[0], np.array([0.1, 0.5]), path[-1]]
        monkeypatch.setattr(planner, "_try_shortcut", lambda p, a, b: list(detour))
        planner._rng = np.random.default_rng(0)
        assert [list(q) for q in planner._smooth_path(prob, list(path))] == [list(q) for q in path]

        # A shortcut of equal length is not an improvement either
        same = [path[0], np.array([0.1, -0.1]), path[-1]]
        monkeypatch.setattr(planner, "_try_shortcut", lambda p, a, b: list(same))
        planner._rng = np.random.default_rng(0)
        assert [list(q) for q in planner._smooth_path(prob, list(path))] == [list(q) for q in path]

        # A genuinely shorter one is accepted
        direct = [path[0], path[-1]]
        monkeypatch.setattr(planner, "_try_shortcut", lambda p, a, b: list(direct))
        planner._rng = np.random.default_rng(0)
        assert len(planner._smooth_path(prob, list(path))) == 2

    def test_smoothing_reduces_length_not_just_count(self):
        planner = make_planner(step_size=0.3, smoothing_iterations=200)
        path = self.zigzag()
        space = planner.space
        before = sum(space.distance(a, b) for a, b in zip(path[:-1], path[1:]))
        planner._rng = np.random.default_rng(1)
        smoothed = planner._smooth_path(problem(planner, path[0], path[-1]), list(path))
        after = sum(space.distance(a, b) for a, b in zip(smoothed[:-1], smoothed[1:]))
        assert after <= before + 1e-9

    def test_solve_with_smoothing_keeps_validated_path(self):
        disk = Disk([0.5, 0.25], 0.15)
        planner = make_planner(disk, step_size=0.2, edge_resolution=0.02, connection_tolerance=0.05)
        q0, q1 = np.zeros(2), np.array([1.0, 0.5])
        for seed in range(4):
            result = planner.solve(problem(planner, q0, q1), seed=seed)
            assert result.success
            assert np.array_equal(result.path[0], q0) and np.array_equal(result.path[-1], q1)
            assert_path_validated(planner, result.path)


class TestAngularJointsKeepExactEndpoints:
    def test_goal_representation_is_preserved(self):
        """Even with angular joints, the returned goal is the configuration as given."""
        planner = make_planner(angular_joints=(True, True), step_size=0.2)
        q0, q1 = np.array([3.0, 0.0]), np.array([-3.0, 0.2])
        result = planner.solve(problem(planner, q0, q1), seed=0)
        assert result.success
        assert np.array_equal(result.path[0], q0) and np.array_equal(result.path[-1], q1)
