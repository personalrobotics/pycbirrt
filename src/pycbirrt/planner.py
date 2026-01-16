import numpy as np
from tsr import TSR
from tsr.sampling import sample_from_tsrs

from pycbirrt.config import CBiRRTConfig
from pycbirrt.tree import RRTree
from pycbirrt.interfaces import RobotModel, IKSolver, CollisionChecker


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
    ) -> list[np.ndarray] | None:
        """Plan a path from start configuration to a goal TSR.

        Args:
            start: Start joint configuration
            goal_tsrs: List of TSRs defining the goal region
            seed: Random seed for reproducibility

        Returns:
            List of joint configurations forming the path, or None if no path found
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Validate start
        if not self.collision.is_valid(start):
            raise ValueError("Start configuration is in collision")

        # Sample a goal configuration from the TSRs
        goal_config = self._sample_goal_config(goal_tsrs)
        if goal_config is None:
            return None  # Could not find valid goal config

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

            # Extend tree_a toward sample
            new_idx = self._extend(tree_a, q_sample)
            if new_idx is None:
                continue

            # Try to connect tree_b to the new node
            q_new = tree_a.nodes[new_idx].config
            connect_idx = self._connect(tree_b, q_new)

            if connect_idx is not None:
                # Found a path - extract and return it
                path = self._extract_path(
                    tree_start, tree_goal, tree_a, tree_b, new_idx, connect_idx
                )
                if self.config.smooth_path:
                    path = self._smooth_path(path)
                return path

        return None  # No path found within iteration limit

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

    def _extend(self, tree: RRTree, q_target: np.ndarray) -> int | None:
        """Extend tree toward target configuration.

        Args:
            tree: Tree to extend
            q_target: Target configuration

        Returns:
            Index of new node, or None if extension failed
        """
        # Find nearest node
        nearest_idx = tree.nearest(q_target)
        q_near = tree.nodes[nearest_idx].config

        # Compute direction and step
        direction = q_target - q_near
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            return None

        # Limit step size
        if distance > self.config.step_size:
            direction = direction / distance * self.config.step_size

        q_new = q_near + direction

        # Check validity
        if not self._is_within_limits(q_new):
            return None
        if not self.collision.is_valid(q_new):
            return None
        if not self._is_edge_valid(q_near, q_new):
            return None

        return tree.add_node(q_new, nearest_idx)

    def _connect(self, tree: RRTree, q_target: np.ndarray) -> int | None:
        """Try to connect tree to target configuration.

        Args:
            tree: Tree to connect from
            q_target: Target configuration to connect to

        Returns:
            Index of node that reached target, or None if connection failed
        """
        nearest_idx = tree.nearest(q_target)

        for _ in range(self.config.max_connect_attempts):
            q_near = tree.nodes[nearest_idx].config
            direction = q_target - q_near
            distance = np.linalg.norm(direction)

            if distance < self.config.goal_tolerance:
                return nearest_idx  # Close enough

            # Step toward target
            if distance > self.config.connect_step_size:
                direction = direction / distance * self.config.connect_step_size

            q_new = q_near + direction

            # Check validity
            if not self._is_within_limits(q_new):
                return None
            if not self.collision.is_valid(q_new):
                return None
            if not self._is_edge_valid(q_near, q_new):
                return None

            nearest_idx = tree.add_node(q_new, nearest_idx)

        return None

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
