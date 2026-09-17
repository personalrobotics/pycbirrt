# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import logging
import time
from dataclasses import dataclass

import numpy as np
from tsr import TSR, Constraint, choose_tsr_index
from tsr.bimanual import BimanualPose
from tsr.sampling import sample_from_tsrs

from pycbirrt.config import CBiRRTConfig
from pycbirrt.exceptions import (
    AllGoalConfigurationsInCollision,
    AllGoalConfigurationsInvalid,
    AllStartConfigurationsInCollision,
    AllStartConfigurationsInvalid,
)
from pycbirrt.interfaces import CollisionChecker, IKSolver, RobotModel
from pycbirrt.space import JointSpace
from pycbirrt.tree import RRTree

logger = logging.getLogger(__name__)


def _merge_bimanual_poses(primary, secondary):
    """Fold ``secondary``'s set component into ``primary`` wherever ``primary``
    leaves it unset (``None``).

    A goal-region TSR and a path-constraint TSR are typically defined over
    disjoint components (e.g. absolute vs. relative for a bimanual pose), so
    their samples can be solved as a single combined IK target instead of
    solving the goal alone and rejecting it after the fact if the constraint
    happens not to hold. A no-op for pose types (e.g. a plain single-arm
    Motor) that don't decompose this way.
    """
    if not isinstance(primary, BimanualPose) or not isinstance(secondary, BimanualPose):
        return primary
    return BimanualPose(
        absolute=primary.absolute if primary.absolute is not None else secondary.absolute,
        relative=primary.relative if primary.relative is not None else secondary.relative,
    )


@dataclass
class PlanResult:
    """Result of a planning query.

    Attributes:
        path: Joint configurations from start to goal, or None if failed.
        start_index: Index into the start input (config list or TSR list)
            that the path begins from. Always 0 for single start.
        goal_index: Index into the goal input (config list or TSR list)
            that the path ends at. Always 0 for single goal.
        iterations: Number of RRT iterations used.
        planning_time: Wall-clock time in seconds.
        tree_sizes: Tuple of (start_tree_size, goal_tree_size).
        success: Whether a path was found.
        failure_reason: Human-readable reason for failure, or None if success.
    """

    path: list[np.ndarray] | None
    start_index: int
    goal_index: int
    iterations: int
    planning_time: float
    tree_sizes: tuple[int, int]
    success: bool
    failure_reason: str | None = None


class CBiRRT:
    """Constrained Bi-directional RRT planner with TSR constraints.

    Supports three types of TSR constraints:
    - start_tsrs: Define valid start regions (optional, use fixed start if not provided)
    - goal_tsrs: Define valid goal regions
    - constraint_tsrs: Path constraints that must be satisfied along the entire trajectory
    """

    def __init__(
        self,
        robot: RobotModel,
        ik_solver: IKSolver,
        collision_checker: CollisionChecker,
        config: CBiRRTConfig | None = None,
    ):
        """Initialize the CBiRRT planner.

        Args:
            robot: Robot model providing FK and joint limits
            ik_solver: Inverse kinematics solver
            collision_checker: Collision checking interface
            config: Planner configuration (uses defaults if None)
        """
        self.robot = robot
        self.ik = ik_solver
        self.collision = collision_checker
        self.config = config or CBiRRTConfig()

        # Joint-space geometry: limits, metric, interpolation, sampling.
        # Raises ValueError if angular_joints length does not match robot DOF.
        lower, upper = robot.joint_limits
        self.space = JointSpace(lower, upper, angular_joints=self.config.angular_joints)

        self._rng = np.random.default_rng()
        # Running upper bound on the metric volume element, for rejection
        # sampling; it only ever grows, so it self-corrects from below.
        self._sample_density_bound = 0.0
        # Path constraints are queried only through the Constraint seam
        # (distance / to_transform), so any Constraint works — TSR or a geometric
        # primitive such as PlaneConstraint. Start/goal regions additionally need
        # volume-weighted sampling (.sample + .Bw), which today only TSR provides.
        self._constraint_tsrs: list[Constraint] | None = None
        self._start_tsrs: list[TSR] | None = None
        self._goal_tsrs: list[TSR] | None = None

    def plan(
        self,
        start: np.ndarray | list[np.ndarray] | None = None,
        goal: np.ndarray | list[np.ndarray] | None = None,
        goal_tsrs: list[TSR] | None = None,
        start_tsrs: list[TSR] | None = None,
        constraint_tsrs: list[Constraint] | None = None,
        seed: int | None = None,
        return_details: bool = False,
    ) -> list[np.ndarray] | None | PlanResult:
        """Plan a path from start to goal with optional TSR constraints.

        Multiple TSRs in goal_tsrs or start_tsrs are treated as a union - the
        planner will find a path to ANY of the goal TSRs. TSRs are sampled
        proportionally to their volume (sum of Bw bounds), so TSRs with more
        freedom are explored more frequently.

        Multiple discrete configurations can be provided as lists - all become
        tree roots and are explored simultaneously.

        Args:
            start: Start configuration(s). Can be:
                   - Single config: np.ndarray
                   - Multiple configs: list[np.ndarray]
                   - None (must provide start_tsrs)
            goal: Goal configuration(s). Can be:
                  - Single config: np.ndarray
                  - Multiple configs: list[np.ndarray]
                  - None (must provide goal_tsrs)
            goal_tsrs: Optional TSRs defining goal region(s) (union).
                      Each TSR is sampled proportionally to its volume.
            start_tsrs: Optional TSRs defining valid start regions (union).
            constraint_tsrs: Optional TSRs that constrain the entire path.
                            Every configuration along the path must satisfy ALL of these.
            seed: Random seed for reproducibility
            return_details: If True, return PlanResult with trees; otherwise just path

        Returns:
            If return_details=False: List of joint configurations or None
            If return_details=True: PlanResult with path, trees, and debug info

        Examples:
            # Single to single
            path = planner.plan(start=q1, goal=q2)

            # Multiple starts to single goal
            path = planner.plan(start=[q1, q2, q3], goal=q_goal)

            # Mix configs and TSRs
            path = planner.plan(start=[q1], goal_tsrs=[tsr1, tsr2])
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Store TSRs for use during planning
        self._constraint_tsrs = constraint_tsrs
        self._start_tsrs = start_tsrs
        self._goal_tsrs = goal_tsrs

        # Normalize start/goal to lists
        start_configs = None
        if start is not None:
            start_configs = [start] if isinstance(start, np.ndarray) else start

        goal_configs = None
        if goal is not None:
            goal_configs = [goal] if isinstance(goal, np.ndarray) else goal

        # Initialize start configurations
        # Validation errors (invalid configs) should propagate up
        # Only sampling failures return None
        start_roots, start_source_indices = self._initialize_tree_configs(start_configs, start_tsrs, "Start")

        # Initialize goal configurations
        goal_roots, goal_source_indices = self._initialize_tree_configs(goal_configs, goal_tsrs, "Goal")

        # Initialize trees with multiple roots and source indices
        tree_start = RRTree(
            start_roots if len(start_roots) > 1 else start_roots[0],
            source_indices=start_source_indices,
        )
        tree_goal = RRTree(
            goal_roots if len(goal_roots) > 1 else goal_roots[0],
            source_indices=goal_source_indices,
        )

        # Track start time for timeout
        start_time = time.monotonic()

        def _make_result(path, iteration, success, failure_reason=None):
            elapsed = time.monotonic() - start_time
            si = tree_start.get_root_source_index(0) if success else 0
            gi = tree_goal.get_root_source_index(0) if success else 0
            return PlanResult(
                path=path,
                start_index=si,
                goal_index=gi,
                iterations=iteration,
                planning_time=elapsed,
                tree_sizes=(len(tree_start), len(tree_goal)),
                success=success,
                failure_reason=failure_reason,
            )

        # Main planning loop
        for iteration in range(self.config.max_iterations):
            # Check abort
            if self.config.abort_fn is not None and self.config.abort_fn():
                reason = "Aborted by user"
                if return_details:
                    return _make_result(None, iteration, False, failure_reason=reason)
                return None

            # Check timeout
            if time.monotonic() - start_time > self.config.timeout:
                reason = (
                    f"Timeout after {self.config.timeout:.1f}s, {iteration} iterations. "
                    f"Trees: start={len(tree_start)} nodes, goal={len(tree_goal)} nodes. "
                    f"Trees could not connect."
                )
                if return_details:
                    return _make_result(None, iteration, False, failure_reason=reason)
                return None

            # Alternate which tree we extend
            if iteration % 2 == 0:
                tree_a, tree_b = tree_start, tree_goal
            else:
                tree_a, tree_b = tree_goal, tree_start

            # Sample configuration (with bias toward opposite tree's TSR)
            q_sample = None
            if tree_a is tree_start and self._goal_tsrs is not None and self._rng.random() < self.config.goal_bias:
                q_sample = self._sample_from_tsrs(self._goal_tsrs, must_satisfy_constraints=True)
            elif tree_a is tree_goal and self._start_tsrs is not None and self._rng.random() < self.config.start_bias:
                q_sample = self._sample_from_tsrs(self._start_tsrs, must_satisfy_constraints=True)
            if q_sample is None:
                q_sample = self._sample_random_config()

            # Extend tree_a toward random sample (uses extend_steps)
            grow_idx, _ = self._grow(tree_a, q_sample, self.config.extend_steps)

            # Connect tree_b toward where tree_a reached (uses connect_steps)
            q_reached = tree_a.nodes[grow_idx].config
            connect_idx, connected = self._grow(tree_b, q_reached, self.config.connect_steps)

            if connected:
                # Found a path - extract and return it
                path = self._extract_path(tree_start, tree_goal, tree_a, tree_b, grow_idx, connect_idx)
                if self.config.smooth_path:
                    path = self._smooth_path(path)

                # Determine which start/goal was used by tracing to root
                if tree_a is tree_start:
                    si = tree_start.get_root_source_index(grow_idx)
                    gi = tree_goal.get_root_source_index(connect_idx)
                else:
                    si = tree_start.get_root_source_index(connect_idx)
                    gi = tree_goal.get_root_source_index(grow_idx)

                elapsed = time.monotonic() - start_time
                if return_details:
                    return PlanResult(
                        path=path,
                        start_index=si if si is not None else 0,
                        goal_index=gi if gi is not None else 0,
                        iterations=iteration + 1,
                        planning_time=elapsed,
                        tree_sizes=(len(tree_start), len(tree_goal)),
                        success=True,
                    )
                return path

        reason = (
            f"Max iterations ({self.config.max_iterations}) reached. "
            f"Trees: start={len(tree_start)} nodes, goal={len(tree_goal)} nodes. "
            f"Trees could not connect."
        )
        if return_details:
            return _make_result(None, self.config.max_iterations, False, failure_reason=reason)
        return None

    def _sample_from_tsrs(self, tsrs: list[TSR], must_satisfy_constraints: bool = False) -> np.ndarray | None:
        """Sample a single valid configuration from TSRs.

        Tries multiple pose samples until finding one with a valid IK solution.
        Used for goal biasing during planning.

        Args:
            tsrs: List of TSRs to sample from (union)
            must_satisfy_constraints: If True, also check path constraint TSRs

        Returns:
            Valid joint configuration or None
        """
        for _ in range(self.config.tsr_samples):
            pose = sample_from_tsrs(tsrs, self._rng)
            if must_satisfy_constraints and self._constraint_tsrs:
                for constraint_tsr in self._constraint_tsrs:
                    pose = _merge_bimanual_poses(pose, constraint_tsr.sample())
            solutions = self.ik.solve_valid(pose)

            for q in solutions:
                if must_satisfy_constraints and not self._satisfies_constraints(q):
                    continue
                return q

        return None

    def _sample_configs_from_tsrs(
        self, tsrs: list[TSR], target_count: int, must_satisfy_constraints: bool = False
    ) -> tuple[list[np.ndarray], list[int], dict[str, int]]:
        """Sample multiple valid configurations from TSRs for tree initialization.

        Samples poses and collects IK solutions from each (up to max_ik_per_pose)
        until we have enough configs or exhaust the sample budget. Capping solutions
        per pose ensures diversity across different TSR samples.

        Args:
            tsrs: List of TSRs to sample from (union)
            target_count: Target number of configs to collect
            must_satisfy_constraints: If True, also check path constraint TSRs

        Returns:
            Tuple of (configs, tsr_indices, stats) where:
            - configs: valid configurations
            - tsr_indices[i]: index into tsrs that produced configs[i]
            - stats: rejection counts {"ik_failed", "in_collision", "constraint_violated"}
        """
        configs = []
        tsr_indices = []
        stats = {"ik_failed": 0, "in_collision": 0, "constraint_violated": 0}
        max_per_pose = self.config.max_ik_per_pose

        for _ in range(self.config.tsr_samples):
            if len(configs) >= target_count:
                break

            tsr_idx = choose_tsr_index(tsrs, self._rng)
            pose = tsrs[tsr_idx].sample()
            if must_satisfy_constraints and self._constraint_tsrs:
                for constraint_tsr in self._constraint_tsrs:
                    pose = _merge_bimanual_poses(pose, constraint_tsr.sample())
            solutions = self.ik.solve_valid(pose)

            if not solutions:
                stats["ik_failed"] += 1
                continue

            # Cap solutions per pose for diversity
            added_from_pose = 0
            for q in solutions:
                if added_from_pose >= max_per_pose:
                    break
                if not self.collision.is_valid(q):
                    stats["in_collision"] += 1
                    continue
                if must_satisfy_constraints and not self._satisfies_constraints(q):
                    stats["constraint_violated"] += 1
                    continue
                configs.append(q)
                tsr_indices.append(tsr_idx)
                added_from_pose += 1
                if len(configs) >= target_count:
                    break

        return configs, tsr_indices, stats

    def _satisfies_constraints(self, q: np.ndarray) -> bool:
        """Check if configuration satisfies all path constraint TSRs.

        Args:
            q: Joint configuration

        Returns:
            True if all constraints are satisfied (or no constraints exist)
        """
        if self._constraint_tsrs is None:
            return True

        # Normalize FK output through the model's pose token (Motor for
        # single-arm, BimanualPose for bimanual); tsr.distance accepts either.
        pose = self.robot.normalize_pose(self.robot.forward_kinematics(q))
        for tsr in self._constraint_tsrs:
            dist, _ = tsr.distance(pose)
            if dist > self.config.tsr_tolerance:
                return False
        return True

    def _check_config(self, q: np.ndarray) -> tuple[bool, str | None]:
        """Check if a configuration is valid.

        Args:
            q: Configuration to check

        Returns:
            Tuple of (is_valid, reason) where reason is None if valid,
            or a string describing why it's invalid.
        """
        if not self.collision.is_valid(q):
            return False, "in collision"

        if self._constraint_tsrs is not None:
            if not self._satisfies_constraints(q):
                return False, "violates path constraints"

        return True, None

    def _initialize_tree_configs(
        self,
        fixed_configs: list[np.ndarray] | None,
        tsrs: list[TSR] | None,
        config_type: str,  # "Start" or "Goal"
    ) -> tuple[list[np.ndarray], list[int]]:
        """Initialize tree with fixed configs and TSR samples.

        Validates all fixed configurations before proceeding. If some configs
        are invalid but others are valid, logs a warning and continues with
        the valid ones. If ALL configs are invalid, raises an appropriate
        exception with details.

        Args:
            fixed_configs: List of fixed configurations (or None)
            tsrs: List of TSRs to sample from (or None)
            config_type: "Start" or "Goal" for error messages

        Returns:
            Tuple of (valid_configs, source_indices) where source_indices[i]
            is the index into fixed_configs or tsrs that produced config i.
            For fixed configs, the index is the position in the input list.
            For TSR-sampled configs, the index is currently 0 (TSR index
            tracking requires changes to sample_from_tsrs).

        Raises:
            AllStartConfigurationsInCollision: If all start configs are in collision
            AllGoalConfigurationsInCollision: If all goal configs are in collision
            AllStartConfigurationsInvalid: If all start configs are invalid (mixed reasons)
            AllGoalConfigurationsInvalid: If all goal configs are invalid (mixed reasons)
            ValueError: If no configs available at all
        """
        valid_configs = []
        source_indices = []
        invalid_details = []
        all_in_collision = True

        # Check all fixed configs first
        if fixed_configs is not None:
            for i, q in enumerate(fixed_configs):
                is_valid, reason = self._check_config(q)
                if is_valid:
                    valid_configs.append(q)
                    source_indices.append(i)
                else:
                    invalid_details.append(f"{config_type}[{i}]: {reason}")
                    if reason != "in collision":
                        all_in_collision = False

        # If we have valid fixed configs, log warning about invalid ones and continue
        if valid_configs and invalid_details:
            logger.warning(
                f"Filtered {len(invalid_details)} invalid {config_type.lower()} "
                f"configuration(s): {'; '.join(invalid_details)}"
            )

        # Sample from TSRs to fill up to num_tree_roots configs
        tsr_stats = None
        if tsrs is not None:
            must_satisfy = self._constraint_tsrs is not None
            needed = max(0, self.config.num_tree_roots - len(valid_configs))
            if needed > 0:
                sampled, tsr_indices, tsr_stats = self._sample_configs_from_tsrs(
                    tsrs,
                    needed,
                    must_satisfy,
                )
                for q, tsr_idx in zip(sampled, tsr_indices):
                    valid_configs.append(q)
                    source_indices.append(tsr_idx)

        # If we have valid configs, return them
        if valid_configs:
            return valid_configs, source_indices

        # No valid configs - raise appropriate exception
        if fixed_configs is not None and len(fixed_configs) > 0:
            # All fixed configs were invalid
            n_configs = len(fixed_configs)
            if config_type == "Start":
                if all_in_collision:
                    raise AllStartConfigurationsInCollision(n_configs, invalid_details)
                else:
                    raise AllStartConfigurationsInvalid(n_configs, invalid_details)
            else:
                if all_in_collision:
                    raise AllGoalConfigurationsInCollision(n_configs, invalid_details)
                else:
                    raise AllGoalConfigurationsInvalid(n_configs, invalid_details)

        # TSR sampling failed — report why
        if tsr_stats is not None:
            total_attempts = sum(tsr_stats.values())
            if total_attempts > 0:
                details = []
                if tsr_stats["ik_failed"]:
                    details.append(f"{tsr_stats['ik_failed']} IK unreachable")
                if tsr_stats["in_collision"]:
                    details.append(f"{tsr_stats['in_collision']} in collision")
                if tsr_stats["constraint_violated"]:
                    details.append(f"{tsr_stats['constraint_violated']} constraint violated")
                summary = ", ".join(details)

                if tsr_stats["in_collision"] and not tsr_stats["ik_failed"]:
                    Ex = (
                        AllGoalConfigurationsInCollision if config_type == "Goal" else AllStartConfigurationsInCollision
                    )
                    raise Ex(tsr_stats["in_collision"], [summary])
                else:
                    Ex = AllGoalConfigurationsInvalid if config_type == "Goal" else AllStartConfigurationsInvalid
                    raise Ex(total_attempts, [summary])

        # No configs provided at all
        raise ValueError(
            f"No valid {config_type.lower()} configurations available. "
            f"Provide either {config_type.lower()} or {config_type.lower()}_tsrs."
        )

    def _project_to_constraint(self, q: np.ndarray) -> np.ndarray | None:
        """Project configuration onto the constraint manifold.

        Uses iterative projection: repeatedly compute the constraint violation,
        project the end-effector pose onto the TSR (both position and orientation),
        solve IK, and repeat until the configuration satisfies all constraints
        or we give up.

        Args:
            q: Configuration to project

        Returns:
            Projected configuration or None if projection fails
        """
        if self._constraint_tsrs is None:
            return q

        q_current = q.copy()
        prev_dist = float("inf")

        for _ in range(self.config.max_projection_iters):
            # Get current end-effector pose (normalized via the model's pose token)
            pose = self.robot.normalize_pose(self.robot.forward_kinematics(q_current))

            # Find the TSR with the largest violation
            max_dist = 0.0
            worst_tsr = None
            worst_bwopt = None
            for tsr in self._constraint_tsrs:
                dist, bwopt = tsr.distance(pose)
                if dist > max_dist:
                    max_dist = dist
                    worst_tsr = tsr
                    worst_bwopt = bwopt

            # Check if we're on the manifold
            if max_dist <= self.config.tsr_tolerance:
                return q_current

            # Check if we're making progress
            if prev_dist - max_dist < self.config.progress_tolerance:
                return None  # Not converging
            prev_dist = max_dist

            # Project pose onto the worst TSR using bwopt (handles both position and orientation).
            # bwopt is the closest point in the TSR's split Bw coords to the current pose;
            # to_transform maps it to the full world-frame end-effector Motor (T0_w * Tw * Tw_e),
            # the same pose space as forward_kinematics, so it flows straight to the IK solver.
            projected_pose = worst_tsr.to_transform(worst_bwopt)

            # Solve IK for the projected pose (pass q_current as hint for iterative solvers)
            solutions = self.ik.solve(projected_pose, q_init=q_current)
            if not solutions:
                return None  # Can't reach projected pose

            # Find solution closest to current config that's within limits
            best_q = None
            best_dist = float("inf")
            for sol in solutions:
                if not self.space.within_limits(sol):
                    continue
                d = self._angular_distance(q_current, sol)
                if d < best_dist:
                    best_dist = d
                    best_q = sol

            if best_q is None:
                return None  # No valid IK solution within joint limits

            q_current = best_q

        # Exceeded max iterations
        return None

    def _sample_random_config(self) -> np.ndarray:
        """Sample a random configuration within joint limits.

        Draws from the joint-space box (``self.space``). For angular joints the
        limits should cover the working range (e.g. ±2π); the angular distance
        metric handles wrapping.

        With ``metric_sampling`` enabled and a metric that exposes a volume
        element, samples are drawn proportional to ``sqrt(det M(q))`` by
        rejection rather than uniformly in joint coordinates. A uniform sample
        spreads an RRT's Voronoi bias by *Euclidean* volume, which pulls tree
        growth back toward the coordinate geometry however the nearest-neighbour
        metric is defined; weighting by the volume element makes the bias follow
        the metric instead.
        """
        volume_element = getattr(self.config.metric, "volume_element", None)
        if not self.config.metric_sampling or volume_element is None:
            return self.space.sample(self._rng)

        # Rejection sampling against a running estimate of the peak density.
        # The bound self-corrects upward whenever a larger value appears, so a
        # loose initial estimate costs a few extra draws, not correctness.
        best = self._sample_density_bound
        candidate = None
        for _ in range(self.config.metric_sampling_tries):
            candidate = self.space.sample(self._rng)
            density = volume_element(candidate)
            if density > best:
                best = density
                self._sample_density_bound = best
            if best <= 0.0 or self._rng.random() < density / best:
                return candidate
        # Budget exhausted: fall back to the last uniform draw, which keeps the
        # sampler total rather than looping forever in a flat region.
        return candidate

    def _angular_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Distance between configurations, handling angular wraparound.

        Wraparound is resolved by ``self.space`` so the metric only ever sees a
        genuine displacement; the metric then decides how to measure it
        (Euclidean by default, or by kinetic energy when one is configured).
        """
        diff = self._angular_direction(q1, q2)
        return self._metric_norm(q1, diff)

    def _angular_direction(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        """Direction from ``q_from`` to ``q_to``. Thin wrapper over ``self.space``."""
        return self.space.direction(q_from, q_to)

    def _metric_norm(self, q: np.ndarray, dq: np.ndarray) -> float:
        """Length of displacement ``dq`` at ``q`` under the configured metric."""
        if self.config.metric is None:
            return float(np.linalg.norm(dq))
        return float(self.config.metric.norm(q, dq))

    def _metric_scale(self, q: np.ndarray, dq: np.ndarray, target: float,
                      length: float) -> np.ndarray:
        """Shorten ``dq`` to metric length ``target`` (``length`` is its current one)."""
        scale_to_length = getattr(self.config.metric, "scale_to_length", None)
        if scale_to_length is None:
            return dq / length * target
        return np.asarray(scale_to_length(q, dq, target), dtype=float)

    def _extension_step(self, q_current: np.ndarray, q_target: np.ndarray,
                        direction: np.ndarray, distance: float) -> np.ndarray | None:
        """One extension step from ``q_current`` toward ``q_target``.

        Straight-line by default. With ``geodesic_extension`` the direction is
        the metric's Riemannian natural gradient of the squared-distance
        potential -- a discrete geodesic step in the sense of Kyaw & Kelly,
        "Geometry-Aware Sampling-Based Motion Planning on Riemannian Manifolds"
        (Algorithm 1). Under an anisotropic metric that direction is *not* the
        straight line: steepest descent turns away from the heavy directions.

        Returns None when no usable step exists (a degenerate gradient).
        """
        target_length = min(distance, self.config.step_size)
        if self.config.geodesic_extension and self.config.metric is not None:
            natural_gradient = getattr(self.config.metric, "natural_gradient", None)
            if natural_gradient is not None:
                descent = np.asarray(natural_gradient(q_current, q_target), dtype=float)
                # Keep the angular convention of the straight-line direction:
                # wraparound was already resolved there.
                if self.space.angular_joints is not None:
                    descent = self._angular_direction(q_current, q_current + descent)
                length = self._metric_norm(q_current, descent)
                if length <= 1e-12:
                    return None
                return self._metric_scale(q_current, descent, target_length, length)
        return self._metric_scale(q_current, direction, target_length, distance)

    def _nearest_node(self, tree: RRTree, q_target: np.ndarray) -> int:
        """Find nearest node in tree using angular-aware distance.

        Args:
            tree: Tree to search
            q_target: Target configuration

        Returns:
            Index of nearest node
        """
        if self.space.angular_joints is None and self.config.metric is None:
            # Use tree's built-in nearest (faster); it hardcodes the Euclidean
            # norm, so it is only valid when no metric is configured.
            return tree.nearest(q_target)

        # Compute angular-aware distances
        best_idx = 0
        best_dist = float("inf")
        for i, node in enumerate(tree.nodes):
            dist = self._angular_distance(node.config, q_target)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _grow(self, tree: RRTree, q_target: np.ndarray, max_steps: int | None = None) -> tuple[int, bool]:
        """Grow tree toward target using EXT or CON behavior.

        When path constraints are active, each new configuration is projected
        onto the constraint manifold before being added to the tree.

        Args:
            tree: Tree to grow
            q_target: Target configuration to grow toward
            max_steps: Maximum steps (None = CON/unlimited, int = EXT/limited)

        Returns:
            Tuple of (node_index, reached) where:
            - node_index: Index of the furthest node reached toward target
            - reached: True if we reached the target within tsr_tolerance
        """
        # Find nearest node (using angular-aware distance)
        current_idx = self._nearest_node(tree, q_target)
        steps_taken = 0
        prev_distance = float("inf")

        while True:
            q_current = tree.nodes[current_idx].config

            # Compute direction and remaining distance (angular-aware, and in
            # the configured metric so step_size means the same thing here as
            # it does to the nearest-neighbour query).
            direction = self._angular_direction(q_current, q_target)
            distance = self._metric_norm(q_current, direction)

            # Check if we've reached the target
            if distance < self.config.tsr_tolerance:
                return current_idx, True

            # Check if we're making progress toward the target
            if prev_distance - distance < self.config.progress_tolerance:
                break
            prev_distance = distance

            # Check if we've hit EXT step limit
            if max_steps is not None and steps_taken >= max_steps:
                break

            # Choose the step. With geodesic_extension the direction comes from
            # the metric's natural gradient (a discrete geodesic step) rather
            # than the straight line; otherwise it is the straight line toward
            # the target. Either way it is then scaled to metric length
            # step_size -- the midpoint rule is not absolutely homogeneous
            # (scaling dq moves the point where M is sampled), so a plain
            # division can miss the requested length badly.
            step = self._extension_step(q_current, q_target, direction, distance)
            if step is None:
                break
            q_new = q_current + step

            # Check joint limits
            if not self.space.within_limits(q_new):
                break

            # Project onto constraint manifold if constraints exist
            if self._constraint_tsrs is not None:
                q_projected = self._project_to_constraint(q_new)
                if q_projected is None:
                    break
                q_new = q_projected

            # Check collision validity of endpoint (fail-fast before edge check)
            if not self.collision.is_valid(q_new):
                break

            # Check edge validity and add intermediate nodes
            edge_result = self._extend_along_edge(tree, current_idx, q_new)
            current_idx = edge_result[0]
            steps_taken += 1
            if not edge_result[1]:
                # Edge check failed partway through
                break

        return current_idx, False

    def _extend_along_edge(self, tree: RRTree, start_idx: int, q_target: np.ndarray) -> tuple[int, bool]:
        """Extend tree along edge, adding intermediate nodes.

        Checks collision and constraint satisfaction at each step. Adds valid
        intermediate configurations to the tree. Stops at the first invalid
        configuration but keeps all valid ones added so far.

        Note: This uses linear interpolation which is correct for the small
        step sizes used (already within step_size from _grow). Angular
        wraparound is handled at the direction/distance level in _grow.

        Args:
            tree: Tree to extend
            start_idx: Index of starting node in tree
            q_target: Target configuration to extend toward

        Returns:
            Tuple of (final_idx, reached_target) where:
            - final_idx: Index of the last valid node added (or start_idx if none added)
            - reached_target: True if we reached q_target
        """
        q_from = tree.nodes[start_idx].config
        distance = self._angular_distance(q_from, q_target)
        n_steps = max(1, int(np.ceil(distance / self.config.step_size)))

        # Use angular-aware direction for interpolation
        direction = self._angular_direction(q_from, q_target)

        current_idx = start_idx
        for i in range(1, n_steps + 1):
            t = i / n_steps
            q = q_from + t * direction

            # Skip validation for endpoint - already checked in _grow
            if i < n_steps:
                if not self.collision.is_valid(q):
                    return current_idx, False
                if not self._satisfies_constraints(q):
                    return current_idx, False

            # Add this valid intermediate node
            current_idx = tree.add_node(q, current_idx)

        return current_idx, True

    def _extract_path(
        self,
        tree_start: RRTree,
        tree_goal: RRTree,
        tree_a: RRTree,
        tree_b: RRTree,
        idx_a: int,
        idx_b: int,
    ) -> list[np.ndarray]:
        """Extract path from connected trees.

        Args:
            tree_start: Tree rooted at start
            tree_goal: Tree rooted at goal
            tree_a: Tree that was extended (contains idx_a)
            tree_b: Tree that connected (contains idx_b)
            idx_a: Node index in tree_a
            idx_b: Node index in tree_b

        Returns:
            Path from start to goal
        """
        # get_path_to_root returns path from ROOT to the specified node
        if tree_a is tree_start:
            # tree_a=start was extended to idx_a, tree_b=goal connected at idx_b
            path_from_start = tree_start.get_path_to_root(idx_a)  # start -> idx_a
            path_from_goal = tree_goal.get_path_to_root(idx_b)  # goal -> idx_b
            # We want: start -> idx_a -> idx_b -> goal
            # path_from_goal reversed gives: idx_b -> goal
            return path_from_start + list(reversed(path_from_goal))
        else:
            # tree_a=goal was extended to idx_a, tree_b=start connected at idx_b
            path_from_start = tree_start.get_path_to_root(idx_b)  # start -> idx_b
            path_from_goal = tree_goal.get_path_to_root(idx_a)  # goal -> idx_a
            # We want: start -> idx_b -> idx_a -> goal
            # path_from_goal reversed gives: idx_a -> goal
            return path_from_start + list(reversed(path_from_goal))

    def _smooth_path(self, path: list[np.ndarray]) -> list[np.ndarray]:
        """Smooth path using shortcutting with the grow function.

        Picks two random points on the path and attempts to grow from one
        to the other. If successful, replaces the intermediate waypoints
        with the new shorter path segment.

        Stops early if no improvement is made for `smoothing_patience` attempts.

        Args:
            path: Original path

        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path

        smoothed = list(path)
        attempts_without_improvement = 0

        for _ in range(self.config.smoothing_iterations):
            if len(smoothed) <= 2:
                break

            # Early termination if no progress
            if attempts_without_improvement >= self.config.smoothing_patience:
                break

            # Pick two random points (need at least one point between them)
            i = self._rng.integers(0, len(smoothed) - 2)
            j = self._rng.integers(i + 2, len(smoothed))

            # Try to grow from i to j
            shortcut = self._try_shortcut(smoothed[i], smoothed[j])
            improved = False
            if shortcut is not None:
                candidate = smoothed[:i] + shortcut + smoothed[j + 1 :]
                # Accept on *path cost* in the configured metric, not on
                # waypoint count: under a kinetic-energy metric a shortcut with
                # fewer waypoints can still cost more work, and keeping it would
                # undo exactly what the metric was chosen to optimise.
                if self._path_cost(candidate) < self._path_cost(smoothed):
                    smoothed = candidate
                    improved = True

            attempts_without_improvement = 0 if improved else attempts_without_improvement + 1

        return smoothed

    def _path_cost(self, path) -> float:
        """Total length of a path under the configured metric."""
        return float(sum(self._angular_distance(path[k], path[k + 1])
                         for k in range(len(path) - 1)))

    def _try_shortcut(self, q_from: np.ndarray, q_to: np.ndarray) -> list[np.ndarray] | None:
        """Try to find a shorter path between two configurations using grow.

        Args:
            q_from: Start configuration
            q_to: Target configuration

        Returns:
            List of configurations from q_from to q_to (inclusive), or None if failed
        """
        # Create a temporary tree rooted at q_from
        temp_tree = RRTree(q_from)

        # Try to grow toward q_to
        final_idx, reached = self._grow(temp_tree, q_to, max_steps=None)

        if not reached:
            return None

        # Extract path from root to final node
        # get_path_to_root returns root->leaf order
        shortcut = temp_tree.get_path_to_root(final_idx)

        # Replace last point with exact target (grow reaches within tolerance)
        shortcut[-1] = q_to

        return shortcut
