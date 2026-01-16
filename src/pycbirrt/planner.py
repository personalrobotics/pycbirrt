from dataclasses import dataclass
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
    """Constrained Bi-directional RRT planner with TSR goal regions."""

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

    def plan(
        self,
        start: np.ndarray,
        goal_tsrs: list[TSR],
        seed: int | None = None,
        return_details: bool = False,
    ) -> list[np.ndarray] | None | PlanResult:
        """Plan a path from start configuration to a goal TSR.

        Args:
            start: Start joint configuration
            goal_tsrs: List of TSRs defining the goal region
            seed: Random seed for reproducibility
            return_details: If True, return PlanResult with trees; otherwise just path

        Returns:
            If return_details=False: List of joint configurations or None
            If return_details=True: PlanResult with path, trees, and debug info
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Validate start
        if not self.collision.is_valid(start):
            raise ValueError("Start configuration is in collision")

        # Sample a goal configuration from the TSRs
        goal_config = self._sample_goal_config(goal_tsrs)
        if goal_config is None:
            if return_details:
                # Return empty trees for debugging
                tree_start = RRTree(start)
                tree_goal = RRTree(start)  # Dummy
                return PlanResult(None, tree_start, tree_goal, 0, False)
            return None

        # Initialize trees
        tree_start = RRTree(start)
        tree_goal = RRTree(goal_config)

        # Main planning loop
        for iteration in range(self.config.max_iterations):
            # Alternate which tree we extend
            if iteration % 2 == 0:
                tree_a, tree_b = tree_start, tree_goal
            else:
                tree_a, tree_b = tree_goal, tree_start

            # Sample configuration (with goal bias for start tree)
            if tree_a is tree_start and self._rng.random() < self.config.goal_bias:
                q_sample = self._sample_goal_config(goal_tsrs)
                if q_sample is None:
                    q_sample = self._sample_random_config()
            else:
                q_sample = self._sample_random_config()

            # Both trees use same extension behavior (CON if extension_steps=None, else EXT)
            # 1. Grow tree_a toward random sample
            grow_idx, _ = self._grow(tree_a, q_sample)

            # 2. Grow tree_b toward where tree_a reached
            q_reached = tree_a.nodes[grow_idx].config
            connect_idx, connected = self._grow(tree_b, q_reached)

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

    def _sample_goal_config(self, goal_tsrs: list[TSR]) -> np.ndarray | None:
        """Sample a valid configuration from goal TSRs.

        Args:
            goal_tsrs: List of goal TSRs

        Returns:
            Valid joint configuration or None
        """
        for _ in range(100):  # Max attempts
            # Sample pose from TSRs
            pose = sample_from_tsrs(goal_tsrs)

            # Solve IK
            solutions = self.ik.solve(pose)
            if not solutions:
                continue

            # Check each solution for collisions
            for q in solutions:
                if self._is_within_limits(q) and self.collision.is_valid(q):
                    return q

        return None

    def _sample_random_config(self) -> np.ndarray:
        """Sample a random configuration within joint limits."""
        lower, upper = self.robot.joint_limits
        return self._rng.uniform(lower, upper)

    def _is_within_limits(self, q: np.ndarray) -> bool:
        """Check if configuration is within joint limits."""
        lower, upper = self.robot.joint_limits
        return bool(np.all(q >= lower) and np.all(q <= upper))

    def _grow(self, tree: RRTree, q_target: np.ndarray) -> tuple[int, bool]:
        """Grow tree toward target using EXT or CON behavior.

        Extension behavior depends on config.extension_steps:
        - None (CON): Keep stepping until blocked or target reached
        - int X (EXT): Take at most X steps toward target

        Args:
            tree: Tree to grow
            q_target: Target configuration to grow toward

        Returns:
            Tuple of (node_index, reached) where:
            - node_index: Index of the furthest node reached toward target
            - reached: True if we reached the target within goal_tolerance
        """
        # Find nearest node
        current_idx = tree.nearest(q_target)
        steps_taken = 0
        max_steps = self.config.extension_steps  # None = unlimited (CON)

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

            # Check validity - if any check fails, we're blocked
            if not self._is_within_limits(q_new):
                break
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

            # Try to shortcut
            if self._is_edge_valid(smoothed[i], smoothed[j]):
                # Remove intermediate points
                smoothed = smoothed[: i + 1] + smoothed[j:]

        return smoothed
