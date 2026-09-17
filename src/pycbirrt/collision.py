# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Capsule self-collision checking for multi-arm systems.

pycbirrt takes any object with ``is_valid(q) -> bool`` as its collision
checker, and the gafro backends ship no geometry engine of their own. For an
array of arms whose workspaces overlap this is not a detail: without a checker
the planner will happily route one arm straight through another.

This module approximates each link as a **capsule** -- a cylinder with
hemispherical caps, i.e. every point within a radius of the segment joining two
consecutive link origins. Capsules are the natural fit for serial-link arms
(a link *is* a swept segment), the segment-segment distance has a closed form,
and the approximation is conservative if the radius covers the real geometry.

Only pairs from *different* arms are tested by default: consecutive links of one
arm always touch at their shared joint, and a serial arm's self-collisions are a
separate concern from two arms meeting in the middle.

Boxes in the world are handled by the same capsule machinery: a capsule-vs-box
test is the segment's distance to the box minus the capsule radius. Without it
an arm will happily plan straight through the object it is supposed to be
manipulating -- measured at 9 cm of penetration at the grasp and 13 cm along a
lift path before this was added.
"""

from __future__ import annotations

import numpy as np


def segment_distance(p1: np.ndarray, q1: np.ndarray,
                     p2: np.ndarray, q2: np.ndarray) -> float:
    """Shortest distance between segments ``p1->q1`` and ``p2->q2``.

    The standard clamped closed form (Ericson, *Real-Time Collision Detection*,
    §5.1.9): minimise over the two line parameters, then clamp each to [0, 1]
    and re-minimise the other, which handles the parallel and degenerate cases
    without a special branch.
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)

    epsilon = 1e-12
    if a <= epsilon and e <= epsilon:          # both degenerate to points
        return float(np.linalg.norm(p1 - p2))
    if a <= epsilon:                           # first is a point
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e <= epsilon:                       # second is a point
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(d1 @ d2)
            denominator = a * e - b * b
            s = np.clip((b * f - c * e) / denominator, 0.0, 1.0) if denominator > epsilon else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def point_box_distance(point: np.ndarray, centre: np.ndarray,
                       half_extent: np.ndarray) -> float:
    """Signed distance from ``point`` to an axis-aligned box.

    Positive outside, negative inside (the negative branch is the usual
    cheap approximation: the distance to the nearest face).
    """
    offset = np.abs(np.asarray(point, dtype=float) - centre) - half_extent
    outside = float(np.linalg.norm(np.maximum(offset, 0.0)))
    inside = float(min(offset.max(), 0.0))
    return outside + inside


def segment_box_distance(start: np.ndarray, end: np.ndarray, centre: np.ndarray,
                         half_extent: np.ndarray, samples: int = 9) -> float:
    """Smallest signed distance from a segment to an axis-aligned box.

    Sampled along the segment rather than solved exactly: the closed form for
    segment-vs-box is fiddly, and for collision *rejection* a fixed sampling is
    enough as long as it is dense relative to the capsule radius.
    """
    return min(point_box_distance(start + (end - start) * t, centre, half_extent)
               for t in np.linspace(0.0, 1.0, samples))


class Box:
    """An axis-aligned box obstacle."""

    def __init__(self, centre, size):
        self.centre = np.asarray(centre, dtype=float)
        self.half_extent = np.asarray(size, dtype=float) / 2.0

    def distance_to_segment(self, start: np.ndarray, end: np.ndarray) -> float:
        return segment_box_distance(start, end, self.centre, self.half_extent)


class CapsuleSelfCollisionChecker:
    """Rejects configurations where capsules of different arms overlap.

    Args:
        system: the gafro ``System``.
        model: the pycbirrt model, used to widen a planning-width ``q`` to the
            System width its forward kinematics needs.
        link_names: links to build capsules from. Defaults to every link whose
            name is prefixed ``arm<i>/``, grouped by that prefix.
        radius: capsule radius in metres. The default suits a UR5e-sized arm;
            raise it for a safety margin, lower it if the arms must work closer.
        margin: extra clearance required on top of the two radii.
        ground_z: if given, also reject configurations with any capsule below
            this height (the arms are mounted on a table at z = 0).
    """

    def __init__(self, system, model, link_names=None, radius: float = 0.07,
                 margin: float = 0.0, ground_z: float | None = None,
                 obstacles=None, contact_links=None, contact_margin: float = 0.0):
        self.system = system
        self.model = model
        self.radius = float(radius)
        self.margin = float(margin)
        self.ground_z = ground_z
        self.obstacles = list(obstacles or ())
        # Links allowed to touch an obstacle -- the ones doing the grasping.
        # Everything else must stay clear of it by ``contact_margin``.
        self.contact_links = set(contact_links or ())
        self.contact_margin = float(contact_margin)

        names = link_names if link_names is not None else [
            name for name in system.get_link_names() if name.startswith("arm")]
        # Group consecutive links per arm; a capsule spans each adjacent pair.
        self._arms: dict[str, list[str]] = {}
        for name in names:
            self._arms.setdefault(name.split("/")[0], []).append(name)
        self._arm_order = sorted(self._arms)

        task_space = getattr(model, "cooperative", None) or getattr(model, "manipulator", None)
        self._task_indices = (None if task_space is None
                              else np.asarray(task_space.get_joint_indices(), dtype=int))
        self._system_dof = int(system.get_dof())
        self.checks = 0
        self.rejections = 0

    def _system_configuration(self, q: np.ndarray) -> np.ndarray:
        """Planning-width ``q`` -> the System-width vector FK needs."""
        q = np.asarray(q, dtype=float)
        widen = getattr(self.model, "to_system_configuration", None)
        if widen is not None:
            return np.asarray(widen(q), dtype=float)
        to_task_full = getattr(self.model, "_to_task_full", None)
        if to_task_full is not None and self._task_indices is not None:
            system_q = np.zeros(self._system_dof, dtype=float)
            system_q[self._task_indices] = to_task_full(q)
            return system_q
        if len(q) == self._system_dof:
            return q
        system_q = np.zeros(self._system_dof, dtype=float)
        system_q[: len(q)] = q
        return system_q

    def capsules(self, q: np.ndarray, with_names: bool = False):
        """Capsule segments per arm at configuration ``q``.

        With ``with_names`` each segment carries the name of the link it ends
        at, so callers can tell a grasping fingertip from a forearm.
        """
        system_q = self._system_configuration(q)
        origins: dict[str, np.ndarray] = {}
        for arm in self._arm_order:
            for name in self._arms[arm]:
                translator = self.system.compute_link_motor(name, system_q).get_translator()
                origins[name] = np.array([translator.x(), translator.y(), translator.z()],
                                         dtype=float)
        segments: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for arm in self._arm_order:
            links = self._arms[arm]
            arm_segments = []
            for first, second in zip(links, links[1:]):
                start, end = origins[first], origins[second]
                if np.linalg.norm(end - start) > 1e-9:   # skip zero-length links
                    arm_segments.append((start, end, second) if with_names
                                        else (start, end))
            segments[arm] = arm_segments
        return segments

    def obstacle_clearance(self, q: np.ndarray) -> float:
        """Smallest gap between any non-grasping capsule and any obstacle.

        Negative means an arm is inside an obstacle. Links named in
        ``contact_links`` are skipped -- those are the ones meant to be touching.
        """
        if not self.obstacles:
            return float("inf")
        smallest = float("inf")
        for arm_segments in self.capsules(q, with_names=True).values():
            for start, end, link_name in arm_segments:
                if link_name in self.contact_links:
                    continue
                for obstacle in self.obstacles:
                    smallest = min(smallest,
                                   obstacle.distance_to_segment(start, end) - self.radius)
        return smallest

    def clearance(self, q: np.ndarray) -> float:
        """Smallest surface-to-surface gap between capsules of different arms.

        Negative means interpenetration. ``inf`` when there is only one arm.
        """
        segments = self.capsules(q)
        smallest = float("inf")
        for i, arm_a in enumerate(self._arm_order):
            for arm_b in self._arm_order[i + 1:]:
                for start_a, end_a in segments[arm_a]:
                    for start_b, end_b in segments[arm_b]:
                        gap = (segment_distance(start_a, end_a, start_b, end_b)
                               - 2.0 * self.radius)
                        smallest = min(smallest, gap)
        return smallest

    def is_valid(self, q) -> bool:
        """True when nothing overlaps: arm-arm, arm-obstacle, or arm-ground."""
        self.checks += 1
        if self.ground_z is not None:
            for arm_segments in self.capsules(q).values():
                for start, end in arm_segments:
                    if min(start[2], end[2]) - self.radius < self.ground_z:
                        self.rejections += 1
                        return False
        if self.clearance(q) < self.margin:
            self.rejections += 1
            return False
        if self.obstacle_clearance(q) < self.contact_margin:
            self.rejections += 1
            return False
        return True
