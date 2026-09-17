#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Capsule self-collision checking between arms."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pycbirrt.collision import CapsuleSelfCollisionChecker, segment_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from _circle_array import DEFAULT_SOURCE  # noqa: E402

requires_source = pytest.mark.skipif(
    not os.path.exists(DEFAULT_SOURCE), reason=f"missing {DEFAULT_SOURCE}")


class TestSegmentDistance:
    """The closed form must match a brute-force search."""

    @staticmethod
    def brute_force(p1, q1, p2, q2, samples=300):
        t = np.linspace(0.0, 1.0, samples)
        a = p1 + np.outer(t, q1 - p1)
        b = p2 + np.outer(t, q2 - p2)
        return float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)))

    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            points = [rng.normal(size=3) for _ in range(4)]
            exact = segment_distance(*points)
            assert exact == pytest.approx(self.brute_force(*points), abs=5e-3)

    def test_crossing_segments_touch(self):
        distance = segment_distance(np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]),
                                    np.array([0.0, -1, 0]), np.array([0.0, 1, 0]))
        assert distance == pytest.approx(0.0, abs=1e-9)

    def test_parallel_segments(self):
        distance = segment_distance(np.zeros(3), np.array([1.0, 0, 0]),
                                    np.array([0.0, 1, 0]), np.array([1.0, 1, 0]))
        assert distance == pytest.approx(1.0)

    def test_degenerate_points(self):
        distance = segment_distance(np.zeros(3), np.zeros(3),
                                    np.array([3.0, 0, 0]), np.array([3.0, 0, 0]))
        assert distance == pytest.approx(3.0)

    def test_is_symmetric(self):
        rng = np.random.default_rng(1)
        p1, q1, p2, q2 = (rng.normal(size=3) for _ in range(4))
        assert segment_distance(p1, q1, p2, q2) == pytest.approx(
            segment_distance(p2, q2, p1, q1))


@requires_source
class TestCapsuleSelfCollisionChecker:
    @staticmethod
    def _build(arm_count=4, radius=0.7, out_dir=None):
        from circle_array_tsr import build_model, build_system

        system, _path = build_system(arm_count, radius, out_dir)
        model, _label = build_model(arm_count, system)
        return system, model

    def test_seed_configuration_is_clear(self, tmp_path):
        from circle_array_tsr import seed_configuration

        system, model = self._build(out_dir=tmp_path)
        checker = CapsuleSelfCollisionChecker(system, model)
        q = seed_configuration(model, 4)
        assert checker.clearance(q) > 0
        assert checker.is_valid(q)

    def test_rejects_exactly_the_colliding_configurations(self, tmp_path):
        """is_valid must agree with the sign of the clearance."""
        from circle_array_tsr import seed_configuration

        system, model = self._build(out_dir=tmp_path)
        checker = CapsuleSelfCollisionChecker(system, model)
        rng = np.random.default_rng(0)
        start = seed_configuration(model, 4)
        for _ in range(60):
            q = start + rng.normal(0, 0.5, model.dof)
            assert checker.is_valid(q) == (checker.clearance(q) >= 0.0)

    def test_finds_real_collisions(self, tmp_path):
        """The arms face inward, so some nearby configurations really do collide.

        If this ever stops finding any, the example has become a free-space demo
        again and the checker is no longer proving anything.
        """
        from circle_array_tsr import seed_configuration

        system, model = self._build(out_dir=tmp_path)
        checker = CapsuleSelfCollisionChecker(system, model)
        rng = np.random.default_rng(0)
        start = seed_configuration(model, 4)
        clearances = [checker.clearance(start + rng.normal(0, 0.5, model.dof))
                      for _ in range(100)]
        assert min(clearances) < 0.0

    def test_single_arm_has_no_pairs(self, tmp_path):
        from circle_array_tsr import seed_configuration

        system, model = self._build(arm_count=1, out_dir=tmp_path)
        checker = CapsuleSelfCollisionChecker(system, model)
        assert checker.clearance(seed_configuration(model, 1)) == float("inf")
        assert checker.is_valid(seed_configuration(model, 1))

    def test_larger_radius_is_more_conservative(self, tmp_path):
        from circle_array_tsr import seed_configuration

        system, model = self._build(out_dir=tmp_path)
        q = seed_configuration(model, 4)
        thin = CapsuleSelfCollisionChecker(system, model, radius=0.05)
        fat = CapsuleSelfCollisionChecker(system, model, radius=0.12)
        assert fat.clearance(q) < thin.clearance(q)

    def test_margin_requires_extra_clearance(self, tmp_path):
        from circle_array_tsr import seed_configuration

        system, model = self._build(out_dir=tmp_path)
        q = seed_configuration(model, 4)
        checker = CapsuleSelfCollisionChecker(system, model)
        generous = CapsuleSelfCollisionChecker(
            system, model, margin=checker.clearance(q) + 0.1)
        assert checker.is_valid(q)
        assert not generous.is_valid(q)

    def test_planned_paths_stay_clear(self, tmp_path):
        """The whole point: a planned path must not pass arms through each other."""
        from circle_array_tsr import plan_to_goal, seed_configuration

        system, model = self._build(out_dir=tmp_path)
        checker = CapsuleSelfCollisionChecker(system, model)
        start = seed_configuration(model, 4)
        path, _region, _q_goal = plan_to_goal(model, 4, start, attempts=4,
                                              collision=checker)
        assert path is not None
        assert min(checker.clearance(w) for w in path) >= 0.0


class TestBoxDistance:
    """Capsule-vs-box distance, used to keep arms out of the object."""

    def test_point_outside_and_inside(self):
        from pycbirrt.collision import point_box_distance

        centre = np.zeros(3)
        half = np.array([1.0, 1.0, 1.0])
        assert point_box_distance(np.array([2.0, 0, 0]), centre, half) == pytest.approx(1.0)
        assert point_box_distance(np.zeros(3), centre, half) == pytest.approx(-1.0)
        assert point_box_distance(np.array([1.0, 0, 0]), centre, half) == pytest.approx(0.0)

    def test_segment_through_a_box_is_negative(self):
        from pycbirrt.collision import Box

        box = Box(np.zeros(3), np.array([1.0, 1.0, 1.0]))
        assert box.distance_to_segment(np.array([-2.0, 0, 0]), np.array([2.0, 0, 0])) < 0

    def test_segment_beside_a_box_is_positive(self):
        from pycbirrt.collision import Box

        box = Box(np.zeros(3), np.array([1.0, 1.0, 1.0]))
        gap = box.distance_to_segment(np.array([-2.0, 2.0, 0]), np.array([2.0, 2.0, 0]))
        assert gap == pytest.approx(1.5)


@requires_source
class TestObstacleChecking:
    """The object being manipulated must be an obstacle, not scenery."""

    @staticmethod
    def _grasp(tmp_path):
        from circle_array_box_lift import (
            DEFAULT_RADIUS,
            box_from_contacts,
            end_effector_positions,
            grasp_posture,
        )
        from circle_array_tsr import build_model, build_system

        system, _file = build_system(4, DEFAULT_RADIUS, tmp_path)
        model, _label = build_model(4, system)
        q = grasp_posture(model, 4)
        size, centre = box_from_contacts(end_effector_positions(model, system, 4, q))
        return system, model, q, size, centre

    def test_non_gripping_links_stay_out_of_the_box(self, tmp_path):
        """Everything except the fingers must stay clear of the box.

        The gripping links are exempt on purpose: a capsule is a conservative
        hull of the real mesh, so at a genuine grasp (1.8 mm of contact in
        MuJoCo) the capsule model reports ~3 cm of overlap. Enforcing the
        capsule bound on the fingers would make touching the box impossible.
        """
        from circle_array_box_lift import checker_with_box
        from circle_array_tsr import collision_checker

        system, model, q, size, centre = self._grasp(tmp_path)
        checker = checker_with_box(system, model,
                                   collision_checker(system, model), size, centre)
        assert checker.contact_links, "the gripping links should be exempt"
        assert checker.obstacle_clearance(q) >= 0.0, "an arm link reaches into the box"

    def test_gripping_links_really_are_the_ones_that_overlap(self, tmp_path):
        """Without the exemption the fingers are what the capsule model flags."""
        from pycbirrt.collision import Box, CapsuleSelfCollisionChecker

        system, model, q, size, centre = self._grasp(tmp_path)
        strict = CapsuleSelfCollisionChecker(system, model, obstacles=[Box(centre, size)])
        assert strict.obstacle_clearance(q) < 0.0

        box = Box(centre, size)
        offenders = {
            name.split("/")[-1]
            for segments in strict.capsules(q, with_names=True).values()
            for start, end, name in segments
            if box.distance_to_segment(start, end) - strict.radius < 0.0
        }
        assert offenders, "expected the capsule model to flag something"
        assert all(name.startswith("wrist") or name == "ur5e_ee" for name in offenders), (
            f"non-gripping links overlap the box: {offenders}")

    def test_a_box_over_the_arms_is_detected(self, tmp_path):
        """Sanity: a box placed on top of the arms must register as a collision."""
        from pycbirrt.collision import Box, CapsuleSelfCollisionChecker

        system, model, q, _size, _centre = self._grasp(tmp_path)
        swallowing = Box(np.array([0.0, 0.0, 0.4]), np.array([3.0, 3.0, 0.8]))
        checker = CapsuleSelfCollisionChecker(system, model, obstacles=[swallowing])
        assert checker.obstacle_clearance(q) < 0.0
        assert not checker.is_valid(q)

    def test_no_obstacles_means_infinite_clearance(self, tmp_path):
        from pycbirrt.collision import CapsuleSelfCollisionChecker

        system, model, q, _size, _centre = self._grasp(tmp_path)
        checker = CapsuleSelfCollisionChecker(system, model)
        assert checker.obstacle_clearance(q) == float("inf")

    def test_contact_links_may_touch(self, tmp_path):
        """Links named as contacts are exempt -- they are meant to be on the box."""
        from pycbirrt.collision import Box, CapsuleSelfCollisionChecker

        system, model, q, _size, _centre = self._grasp(tmp_path)
        swallowing = Box(np.array([0.0, 0.0, 0.4]), np.array([3.0, 3.0, 0.8]))
        every_link = [n for n in system.get_link_names() if n.startswith("arm")]
        checker = CapsuleSelfCollisionChecker(system, model, obstacles=[swallowing],
                                              contact_links=every_link)
        assert checker.obstacle_clearance(q) == float("inf")
