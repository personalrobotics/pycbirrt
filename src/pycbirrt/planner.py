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

        # Store TSRs for use during planning
        self._constraint_tsrs = constraint_tsrs
        self._start_tsrs = start_tsrs
        self._goal_tsrs = goal_tsrs

        # Determine start configuration
        if start_tsrs is not None:
            # If we have constraint TSRs, sample start that also satisfies them
            must_satisfy = constraint_tsrs is not None
            start_config = self._sample_from_tsrs(start_tsrs, must_satisfy_constraints=must_satisfy)
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
            if time.monotonic() - start_time > self.config.timeout:
                if return_details:
                    return PlanResult(None, tree_start, tree_goal, iteration, False)
                return None

            # Alternate which tree we extend
            if iteration % 2 == 0:
                tree_a, tree_b = tree_start, tree_goal
            else:
                tree_a, tree_b = tree_goal, tree_start

            # Sample configuration (with bias toward opposite tree's TSR)
            q_sample = None
            if tree_a is tree_start and self._rng.random() < self.config.goal_bias:
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
        # Sample pose from TSRs
        pose = sample_from_tsrs(tsrs)

        # Solve IK - returns only valid (collision-free, within limits) solutions
        solutions = self.ik.solve_valid(pose)

        # Check path constraints if required
        for q in solutions:
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
            if dist > self.config.tsr_tolerance:
                return False
        return True

    def _project_to_constraint(self, q: np.ndarray) -> np.ndarray | None:
        """Project configuration onto the constraint manifold.

        Uses iterative projection: repeatedly compute the constraint violation,
        project the end-effector pose onto the TSR, solve IK, and repeat until
        the configuration satisfies all constraints or we give up.

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
            # Get current end-effector pose
            pose = self.robot.forward_kinematics(q_current)

            # Find the TSR with the largest violation
            max_dist = 0.0
            worst_tsr = None
            for tsr in self._constraint_tsrs:
                dist, _ = tsr.distance(pose)
                if dist > max_dist:
                    max_dist = dist
                    worst_tsr = tsr

            # Check if we're on the manifold
            if max_dist <= self.config.tsr_tolerance:
                return q_current

            # Check if we're making progress
            if prev_dist - max_dist < self.config.progress_tolerance:
                return None  # Not converging
            prev_dist = max_dist

            # Project pose onto the worst TSR by clamping to bounds
            tsr = worst_tsr
            T0_w_inv = np.linalg.inv(tsr.T0_w)
            pose_in_tsr = T0_w_inv @ pose

            # Clamp position to TSR bounds
            xyz_in_tsr = pose_in_tsr[:3, 3]
            xyz_clamped = np.clip(xyz_in_tsr, tsr.Bw[:3, 0], tsr.Bw[:3, 1])

            # Create projected pose
            projected_in_tsr = pose_in_tsr.copy()
            projected_in_tsr[:3, 3] = xyz_clamped

            # Transform back to world frame
            projected_pose = tsr.T0_w @ projected_in_tsr

            # Solve IK for the projected pose
            solutions = self.ik.solve(projected_pose)
            if not solutions:
                return None  # Can't reach projected pose

            # Find solution closest to current config that's within limits
            best_q = None
            best_dist = float("inf")
            for sol in solutions:
                if not self._is_within_limits(sol):
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
        """Sample a random configuration within joint limits."""
        lower, upper = self.robot.joint_limits
        return self._rng.uniform(lower, upper)

    def _is_within_limits(self, q: np.ndarray) -> bool:
        """Check if configuration is within joint limits."""
        lower, upper = self.robot.joint_limits
        return bool(np.all(q >= lower) and np.all(q <= upper))

    def _angular_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Compute distance between configurations, handling angular wraparound.

        For angular joints, the distance accounts for the 2*pi wraparound.
        """
        diff = q2 - q1

        if self.config.angular_joints is not None:
            # Wrap angular differences to [-pi, pi]
            for i, is_angular in enumerate(self.config.angular_joints):
                if is_angular:
                    diff[i] = np.arctan2(np.sin(diff[i]), np.cos(diff[i]))

        return float(np.linalg.norm(diff))

    def _nearest_node(self, tree: RRTree, q_target: np.ndarray) -> int:
        """Find nearest node in tree using angular-aware distance.

        Args:
            tree: Tree to search
            q_target: Target configuration

        Returns:
            Index of nearest node
        """
        if self.config.angular_joints is None:
            # Use tree's built-in nearest (faster)
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

    def _angular_direction(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        """Compute direction from q_from to q_to, handling angular wraparound.

        Returns the shortest path direction for angular joints.
        """
        diff = q_to - q_from

        if self.config.angular_joints is not None:
            # Wrap angular differences to [-pi, pi]
            for i, is_angular in enumerate(self.config.angular_joints):
                if is_angular:
                    diff[i] = np.arctan2(np.sin(diff[i]), np.cos(diff[i]))

        return diff

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
            - reached: True if we reached the target within tsr_tolerance
        """
        # Find nearest node (using angular-aware distance)
        current_idx = self._nearest_node(tree, q_target)
        start_idx = current_idx
        prev_distance = float("inf")

        while True:
            q_current = tree.nodes[current_idx].config

            # Compute direction and remaining distance (angular-aware)
            direction = self._angular_direction(q_current, q_target)
            distance = np.linalg.norm(direction)

            # Check if we've reached the target
            if distance < self.config.tsr_tolerance:
                return current_idx, True

            # Check if we're making progress toward the target
            if prev_distance - distance < self.config.progress_tolerance:
                break
            prev_distance = distance

            # Check if we've hit EXT step limit
            if max_steps is not None:
                steps_taken = current_idx - start_idx
                if steps_taken >= max_steps:
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

            # Check collision validity of endpoint
            if not self.collision.is_valid(q_new):
                break

            # Check edge validity and add intermediate nodes
            edge_result = self._extend_along_edge(tree, current_idx, q_new)
            current_idx = edge_result[0]
            if not edge_result[1]:
                # Edge check failed partway through
                break

        return current_idx, False

    def _extend_along_edge(
        self, tree: RRTree, start_idx: int, q_target: np.ndarray
    ) -> tuple[int, bool]:
        """Extend tree along edge, adding intermediate nodes.

        Checks collision and constraint satisfaction at each step. Adds valid
        intermediate configurations to the tree. Stops at the first invalid
        configuration but keeps all valid ones added so far.

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
        distance = np.linalg.norm(q_target - q_from)
        n_steps = max(2, int(np.ceil(distance / self.config.step_size)))

        current_idx = start_idx
        for i in range(1, n_steps + 1):
            t = i / n_steps
            q = q_from + t * (q_target - q_from)

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
        """Smooth path using shortcutting with the grow function.

        Picks two random points on the path and attempts to grow from one
        to the other. If successful, replaces the intermediate waypoints
        with the new shorter path segment.

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

            # Pick two random points (need at least one point between them)
            i = self._rng.integers(0, len(smoothed) - 2)
            j = self._rng.integers(i + 2, len(smoothed))

            # Try to grow from i to j
            shortcut = self._try_shortcut(smoothed[i], smoothed[j])
            if shortcut is not None:
                # Replace path[i:j+1] with the shortcut
                smoothed = smoothed[:i] + shortcut + smoothed[j + 1 :]

        return smoothed

    def _try_shortcut(
        self, q_from: np.ndarray, q_to: np.ndarray
    ) -> list[np.ndarray] | None:
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
