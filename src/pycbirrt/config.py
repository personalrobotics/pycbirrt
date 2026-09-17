# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pycbirrt.metrics import Metric


@dataclass
class CBiRRTConfig:
    """Configuration for CBiRRT planner.

    Extension behavior (EXT vs CON):
    - CON (connect): March until blocked or target reached (extension_steps=None)
    - EXT (extend): Take at most X steps toward target (extension_steps=X)

    The planner supports 4 variants based on extend_steps and connect_steps:
    - CON-CON: Both trees march until blocked (default, like RRT-Connect)
    - EXT-EXT: Both trees take limited steps
    - EXT-CON: Extend tree takes limited steps, connect tree marches
    - CON-EXT: Extend tree marches, connect tree takes limited steps
    """

    # Termination
    timeout: float = 30.0  # Wall-clock timeout in seconds
    max_iterations: int = 100000  # Safety limit (timeout is the primary control)
    tsr_tolerance: float = 1e-3  # Distance tolerance for TSR satisfaction (tree connection + path constraints)
    progress_tolerance: float = 1e-6  # Minimum progress required to continue growing

    # Tree growth parameters
    step_size: float = 0.1  # Maximum joint space step
    goal_bias: float = 0.1  # Probability of start tree sampling from goal TSR
    start_bias: float = 0.1  # Probability of goal tree sampling from start TSR

    # Extension behavior (None = CON, int = EXT with X steps)
    extend_steps: int | None = None  # Steps when growing toward random sample
    connect_steps: int | None = None  # Steps when growing toward other tree

    # Constraint projection
    max_projection_iters: int = 50  # Max iterations for projecting onto constraint manifold

    # TSR sampling
    tsr_samples: int = 100  # Max pose samples to try from TSR
    num_tree_roots: int = 100  # Target number of root configs to seed each tree with
    max_ik_per_pose: int = 3  # Max IK solutions to take per pose sample (for diversity)

    # Smoothing
    smooth_path: bool = True
    smoothing_iterations: int = 50
    smoothing_patience: int = 15  # Stop early if no improvement in this many attempts

    # Angular joints (for proper distance calculation with wraparound)
    # If None, all joints are treated as linear
    # If provided, boolean array where True = angular joint (handles 2*pi wraparound)
    angular_joints: tuple[bool, ...] | None = None

    # Configuration-space metric for nearest-neighbour queries and step sizes.
    # None means the Euclidean norm on joint values (the historical behaviour).
    # Pass a pycbirrt.metrics.KineticEnergyMetric to measure displacements by
    # the work they take (dq^T M(q) dq) instead, so a step that swings the
    # whole arm counts for more than one that flicks the wrist. Note step_size
    # and tsr_tolerance are then in the metric's units, not radians.
    metric: "Metric | None" = None

    # Draw random samples proportional to the metric's volume element
    # sqrt(det M(q)) instead of uniformly in joint coordinates, so the RRT's
    # Voronoi bias follows the metric rather than the coordinates. Ignored when
    # no metric is set, or when the metric exposes no volume element.
    metric_sampling: bool = False
    metric_sampling_tries: int = 16  # rejection draws before falling back

    # Extend along the metric's natural gradient (a discrete geodesic step)
    # instead of the straight line toward the target -- Algorithm 1 of Kyaw &
    # Kelly, "Geometry-Aware Sampling-Based Motion Planning on Riemannian
    # Manifolds". Under an anisotropic metric the two differ: on a UR5e the
    # natural gradient points ~0.91 cosine toward the target where the straight
    # line is 1.0 by construction but is not the cheapest way there. Needs a
    # metric exposing natural_gradient(); ignored otherwise.
    geodesic_extension: bool = False

    # Abort callback — return True to stop planning early
    abort_fn: Callable[[], bool] | None = field(default=None, repr=False)
