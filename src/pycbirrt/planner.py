from dataclasses import dataclass
import time

import numpy as np
from tsr import TSR
from tsr.sampling import sample_from_tsrs

from pycbirrt.config import CBiRRTConfig
from pycbirrt.tree import RRTree
from pycbirrt.interfaces import RobotModel, IKSolver, CollisionChecker


@dataclass
class PlanResult:
    """Result of a planning query."""

    path: list[np.ndarray] | None
    tree_start: RRTree
    tree_goal: RRTree
    iterations: int
    success: bool


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

        self._rng = np.random.default_rng()
        self._constraint_tsrs: list[TSR] | None = None

    def plan(
        self,
        start: np.ndarray | None,
        goal_tsrs: list[TSR],
        start_tsrs: list[TSR] | None = None,
        constraint_tsrs: list[TSR] | None = None,
        seed: int | None = None,
        return_details: bool = False,
    ) -> list[np.ndarray] | None | PlanResult:
        """Plan a path from start to goal with optional TSR constraints.

        Args:
            start: Start joint configuration (required if start_tsrs not provided)
            goal_tsrs: List of TSRs defining the goal region
            start_tsrs: Optional TSRs defining valid start regions
            constraint_tsrs: Optional TSRs that constrain the entire path
            seed: Random seed for reproducibility
            return_details: If True, return PlanResult with trees; otherwise just path

        Returns:
            If return_details=False: List of joint configurations or None
            If return_details=True: PlanResult with path, trees, and debug info
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Store constraint TSRs for use during planning
        self._constraint_tsrs = constraint_tsrs

        # Determine start configuration
        if start_tsrs is not None:
            start_config = self._sample_from_tsrs(start_tsrs)
            if start_config is None:
                if return_details:
                    dummy_tree = RRTree(np.zeros(self.robot.dof))
                    return PlanResult(None, dummy_tree, dummy_tree, 0, False)
                return None
        elif start is not None:
            start_config = start
        else:
            raise ValueError("Either start or start_tsrs must be provided")

        # Validate start
        if not self.collision.is_valid(start_config):
            raise ValueError("Start configuration is in collision")

        # If we have constraint TSRs, verify start satisfies them
        if constraint_tsrs is not None:
            if not self._satisfies_constraints(start_config):
                raise ValueError("Start configuration does not satisfy path constraints")

        # Sample a goal configuration from the TSRs
        goal_config = self._sample_from_tsrs(goal_tsrs, must_satisfy_constraints=True)
        if goal_config is None:
            if return_details:
                tree_start = RRTree(start_config)
                tree_goal = RRTree(start_config)  # Dummy
                return PlanResult(None, tree_start, tree_goal, 0, False)
            return None

        # Initialize trees
        tree_start = RRTree(start_config)
        tree_goal = RRTree(goal_config)

        # Track start time for timeout
        start_time = time.monotonic()

        # Main planning loop
        for iteration in range(self.config.max_iterations):
            # Check timeout
            if self.config.timeout is not None:
                if time.monotonic() - start_time > self.config.timeout:
                    if return_details:
                        return PlanResult(None, tree_start, tree_goal, iteration, False)
                    return None

            # Alternate which tree we extend
            if iteration % 2 == 0:
                tree_a, tree_b = tree_start, tree_goal
            else:
                tree_a, tree_b = tree_goal, tree_start

            # Sample configuration (with goal bias for start tree)
            if tree_a is tree_start and self._rng.random() < self.config.goal_bias:
                q_sample = self._sample_from_tsrs(goal_tsrs, must_satisfy_constraints=True)
                if q_sample is None:
                    q_sample = self._sample_random_config()
            else:
                q_sample = self._sample_random_config()

            # Extend tree_a toward random sample (uses extend_steps)
            grow_idx, _ = self._grow(tree_a, q_sample, self.config.extend_steps)

            # Connect tree_b toward where tree_a reached (uses connect_steps)
            q_reached = tree_a.nodes[grow_idx].config
            connect_idx, connected = self._grow(tree_b, q_reached, self.config.connect_steps)

            if connected:
                # Found a path - extract and return it
                path = self._extract_path(
                    tree_start, tree_goal, tree_a, tree_b, grow_idx, connect_idx
                )
                if self.config.smooth_path:
                    path = self._smooth_path(path)

                if return_details:
                    return PlanResult(path, tree_start, tree_goal, iteration + 1, True)
                return path

        if return_details:
            return PlanResult(None, tree_start, tree_goal, self.config.max_iterations, False)
        return None

    def _sample_from_tsrs(
        self, tsrs: list[TSR], must_satisfy_constraints: bool = False
    ) -> np.ndarray | None:
        """Sample a valid configuration from TSRs.

        Args:
            tsrs: List of TSRs to sample from
            must_satisfy_constraints: If True, also check path constraint TSRs

        Returns:
            Valid joint configuration or None
        """
        for _ in range(100):  # Max attempts
            # Sample pose from TSRs
            pose = sample_from_tsrs(tsrs)

            # Solve IK
            solutions = self.ik.solve(pose)
            if not solutions:
                continue

            # Check each solution for validity
            for q in solutions:
                if not self._is_within_limits(q):
                    continue
                if not self.collision.is_valid(q):
                    continue
                if must_satisfy_constraints and not self._satisfies_constraints(q):
                    continue
                return q

        return None

    def _satisfies_constraints(self, q: np.ndarray) -> bool:
        """Check if configuration satisfies all path constraint TSRs.

        Args:
            q: Joint configuration

        Returns:
            True if all constraints are satisfied (or no constraints exist)
        """
        if self._constraint_tsrs is None:
            return True

        pose = self.robot.forward_kinematics(q)
        for tsr in self._constraint_tsrs:
            dist, _ = tsr.distance(pose)
            if dist > self.config.constraint_tolerance:
                return False
        return True

    def _project_to_constraint(self, q: np.ndarray) -> np.ndarray | None:
        """Project configuration onto the constraint manifold.

        Uses iterative IK to find a nearby configuration that satisfies
        all path constraint TSRs.

        Args:
            q: Configuration to project

        Returns:
            Projected configuration or None if projection fails
        """
        if self._constraint_tsrs is None:
            return q

        # Get current end-effector pose
        pose = self.robot.forward_kinematics(q)

        # Check if already on manifold
        max_dist = 0.0
        for tsr in self._constraint_tsrs:
            dist, _ = tsr.distance(pose)
            max_dist = max(max_dist, dist)

        if max_dist <= self.config.constraint_tolerance:
            return q

        # Project pose onto TSR and solve IK
        # Use the TSR's project method to get the nearest valid pose
        for tsr in self._constraint_tsrs:
            projected_pose = tsr.project(pose)

            # Try to find IK solution close to current config
            solutions = self.ik.solve(projected_pose)
            if not solutions:
                continue

            # Find solution closest to q
            best_q = None
            best_dist = float("inf")
            for sol in solutions:
                if not self._is_within_limits(sol):
                    continue
                dist = np.linalg.norm(sol - q)
                if dist < best_dist:
                    best_dist = dist
                    best_q = sol

            if best_q is not None and self.collision.is_valid(best_q):
                return best_q

        return None

    def _sample_random_config(self) -> np.ndarray:
        """Sample a random configuration within joint limits."""
        lower, upper = self.robot.joint_limits
        return self._rng.uniform(lower, upper)

    def _is_within_limits(self, q: np.ndarray) -> bool:
        """Check if configuration is within joint limits."""
        lower, upper = self.robot.joint_limits
        return bool(np.all(q >= lower) and np.all(q <= upper))

    def _grow(
        self, tree: RRTree, q_target: np.ndarray, max_steps: int | None = None
    ) -> tuple[int, bool]:
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
            - reached: True if we reached the target within goal_tolerance
        """
        # Find nearest node
        current_idx = tree.nearest(q_target)
        steps_taken = 0

        while True:
            q_current = tree.nodes[current_idx].config

            # Compute direction and remaining distance
            direction = q_target - q_current
            distance = np.linalg.norm(direction)

            # Check if we've reached the target
            if distance < self.config.goal_tolerance:
                return current_idx, True

            # Check if we've hit EXT step limit
            if max_steps is not None and steps_taken >= max_steps:
                break

            # Normalize and limit step size
            step = direction / distance * min(distance, self.config.step_size)
            q_new = q_current + step

            # Check joint limits
            if not self._is_within_limits(q_new):
                break

            # Project onto constraint manifold if constraints exist
            if self._constraint_tsrs is not None:
                q_projected = self._project_to_constraint(q_new)
                if q_projected is None:
                    break
                q_new = q_projected

            # Check collision validity
            if not self.collision.is_valid(q_new):
                break
            if not self._is_edge_valid(q_current, q_new):
                break

            # Add node and continue marching
            current_idx = tree.add_node(q_new, current_idx)
            steps_taken += 1

        return current_idx, False

    def _is_edge_valid(
        self, q_from: np.ndarray, q_to: np.ndarray, resolution: float = 0.02
    ) -> bool:
        """Check if edge between two configurations is collision-free.

        Args:
            q_from: Start configuration
            q_to: End configuration
            resolution: Interpolation resolution

        Returns:
            True if edge is collision-free
        """
        distance = np.linalg.norm(q_to - q_from)
        n_steps = max(2, int(np.ceil(distance / resolution)))

        for i in range(1, n_steps):
            t = i / n_steps
            q = q_from + t * (q_to - q_from)
            if not self.collision.is_valid(q):
                return False

        return True

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
            path_from_goal = tree_goal.get_path_to_root(idx_b)    # goal -> idx_b
            # We want: start -> idx_a -> idx_b -> goal
            # path_from_goal reversed gives: idx_b -> goal
            return path_from_start + list(reversed(path_from_goal))
        else:
            # tree_a=goal was extended to idx_a, tree_b=start connected at idx_b
            path_from_start = tree_start.get_path_to_root(idx_b)  # start -> idx_b
            path_from_goal = tree_goal.get_path_to_root(idx_a)    # goal -> idx_a
            # We want: start -> idx_b -> idx_a -> goal
            # path_from_goal reversed gives: idx_a -> goal
            return path_from_start + list(reversed(path_from_goal))

    def _smooth_path(self, path: list[np.ndarray]) -> list[np.ndarray]:
        """Smooth path using shortcutting.

        When path constraints are active, shortcuts must also satisfy
        the constraint TSRs along the interpolated edge.

        Args:
            path: Original path

        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path

        smoothed = list(path)

        for _ in range(self.config.smoothing_iterations):
            if len(smoothed) <= 2:
                break

            # Pick two random points
            i = self._rng.integers(0, len(smoothed) - 1)
            j = self._rng.integers(i + 1, len(smoothed))

            # Try to shortcut (must be collision-free and satisfy constraints)
            if self._is_edge_valid(smoothed[i], smoothed[j]):
                if self._constraint_tsrs is None or self._is_edge_constrained(
                    smoothed[i], smoothed[j]
                ):
                    # Remove intermediate points
                    smoothed = smoothed[: i + 1] + smoothed[j:]

        return smoothed

    def _is_edge_constrained(
        self, q_from: np.ndarray, q_to: np.ndarray, resolution: float = 0.02
    ) -> bool:
        """Check if edge satisfies path constraints along its length.

        Args:
            q_from: Start configuration
            q_to: End configuration
            resolution: Interpolation resolution

        Returns:
            True if all interpolated points satisfy constraints
        """
        distance = np.linalg.norm(q_to - q_from)
        n_steps = max(2, int(np.ceil(distance / resolution)))

        for i in range(1, n_steps):
            t = i / n_steps
            q = q_from + t * (q_to - q_from)
            if not self._satisfies_constraints(q):
                return False

        return True
