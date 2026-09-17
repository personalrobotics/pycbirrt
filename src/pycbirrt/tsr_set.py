# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The configuration-space set induced by a Task Space Region.

A TSR is a set of end-effector poses. Through forward kinematics it induces
a set of configurations, ``{q : FK(q) in TSR}``. This module adapts a TSR to
the state-set protocols in ``pycbirrt.sets`` so the planner can treat it
like any other set. It is the only planner module that imports from the
``tsr`` package besides the legacy argument lowering.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from tsr import wrap_to_interval
from tsr.sampling import weights_from_tsrs

from pycbirrt.interfaces import IKSolver, RobotModel
from pycbirrt.sets import Sample
from pycbirrt.space import JointSpace


class TSRConfigurationSet:
    """Configurations whose end-effector pose lies in a TSR.

    Capabilities:
        contains: TSR distance of ``FK(q)`` is within ``tolerance``.
        distance: the TSR distance of ``FK(q)`` (Berenson et al. 2011, Sec. 4.2).
        sample: draw a pose uniformly from the TSR bounds, solve IK, and
            return one solution within joint limits. With
            ``max_solutions_per_pose > 1``, subsequent calls hand out the
            remaining IK solutions of the same pose before drawing a new one,
            which trades pose diversity for fewer IK calls.
        project: iteratively move the pose to the closest point of the TSR
            and solve IK, choosing the solution nearest the current
            configuration under the joint-space metric. Gives up when the
            violation stops decreasing by ``progress_tolerance`` or after
            ``max_projection_iters``. ``q_previous`` is currently unused.

    Collision is not part of this set. Samples and projections are filtered
    by joint limits only; the planner applies its validator.
    """

    def __init__(
        self,
        tsr: TSR,
        robot: RobotModel,
        ik: IKSolver,
        space: JointSpace,
        tolerance: float = 1e-3,
        max_projection_iters: int = 50,
        progress_tolerance: float = 1e-6,
        max_solutions_per_pose: int = 1,
    ):
        if max_solutions_per_pose < 1:
            raise ValueError("max_solutions_per_pose must be at least 1")
        self.tsr = tsr
        self.robot = robot
        self.ik = ik
        self.space = space
        self.tolerance = tolerance
        self.max_projection_iters = max_projection_iters
        self.progress_tolerance = progress_tolerance
        self.max_solutions_per_pose = max_solutions_per_pose
        self._pending: list[np.ndarray] = []

    def __repr__(self) -> str:
        return f"TSRConfigurationSet({self.tsr!r}, tolerance={self.tolerance})"

    # -- membership and distance ------------------------------------------

    def distance(self, q: np.ndarray) -> float:
        dist, _ = self.tsr.distance(self.robot.forward_kinematics(q))
        return float(dist)

    def contains(self, q: np.ndarray) -> bool:
        return self.distance(q) <= self.tolerance

    # -- sampling ------------------------------------------------------------

    def sample_pose(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a world-frame end-effector pose uniformly from the TSR bounds.

        Uses the caller's generator rather than ``TSR.sample`` so results are
        reproducible under a seed (see personalrobotics/tsr#52).
        """
        bounds = self.tsr._Bw_cont  # continuous bounds: handles wrapped angular intervals
        bw = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * rng.random(6)
        # bw is Motor.log order: rotor bivector first, translation last.
        bw[0:3] = wrap_to_interval(bw[0:3])
        return self.tsr.to_transform(bw)

    def sample(self, rng: np.random.Generator) -> Sample | None:
        if self._pending:
            return Sample(self._pending.pop(0))
        pose = self.sample_pose(rng)
        solutions = [q for q in self.ik.solve(pose) if self.space.within_limits(q)]
        if not solutions:
            return None
        first, rest = solutions[0], solutions[1 : self.max_solutions_per_pose]
        self._pending = [np.array(q, dtype=float) for q in rest]
        return Sample(np.array(first, dtype=float))

    # -- projection ----------------------------------------------------------

    def project(self, q_previous: np.ndarray, q_proposed: np.ndarray) -> np.ndarray | None:
        q = np.array(q_proposed, dtype=float)
        prev_dist = float("inf")
        for _ in range(self.max_projection_iters):
            dist, bwopt = self.tsr.distance(self.robot.forward_kinematics(q))
            if dist <= self.tolerance:
                return q
            if prev_dist - dist < self.progress_tolerance:
                return None
            prev_dist = dist

            target = self.tsr.to_transform(bwopt)
            best, best_d = None, float("inf")
            for sol in self.ik.solve(target, q_init=q):
                if not self.space.within_limits(sol):
                    continue
                d = self.space.distance(q, sol)
                if d < best_d:
                    best, best_d = sol, d
            if best is None:
                return None
            q = np.array(best, dtype=float)
        return None


def tsr_weights(sets: Sequence[TSRConfigurationSet]) -> np.ndarray:
    """Mixture weights for an ``AnyOf`` of TSR sets, proportional to TSR volume.

    Matches the legacy planner's policy: the sum of bound widths, falling
    back to uniform when every TSR has zero volume.
    """
    return weights_from_tsrs([s.tsr for s in sets])
