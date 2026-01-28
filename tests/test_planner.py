"""Tests for CBiRRT planner."""

import numpy as np
import pytest
from tsr import TSR

from pycbirrt import CBiRRT, CBiRRTConfig, AllStartConfigurationsInCollision
from pycbirrt.tree import RRTree


class MockRobotModel:
    """Simple 2-DOF planar arm for testing."""

    def __init__(self):
        self.l1 = 1.0  # Link 1 length
        self.l2 = 1.0  # Link 2 length

    @property
    def dof(self) -> int:
        return 2

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi])

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """2D planar FK embedded in 3D."""
        x = self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1])
        y = self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1])

        T = np.eye(4)
        T[0, 3] = x
        T[1, 3] = y
        return T


class MockCollisionChecker:
    """Always returns valid (no obstacles)."""

    def is_valid(self, q: np.ndarray) -> bool:
        return True

    def is_valid_batch(self, qs: np.ndarray) -> np.ndarray:
        return np.ones(len(qs), dtype=bool)


class MockIKSolver:
    """Analytical IK for 2-DOF planar arm."""

    def __init__(
        self,
        robot: MockRobotModel | None = None,
        collision_checker: MockCollisionChecker | None = None,
    ):
        self.robot = robot or MockRobotModel()
        self.collision = collision_checker or MockCollisionChecker()

    def solve(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Return all kinematic IK solutions (unvalidated).

        Note: q_init is ignored (analytical solver finds all solutions).
        """
        x, y = pose[0, 3], pose[1, 3]
        d = np.sqrt(x**2 + y**2)

        # Check reachability
        if d > self.robot.l1 + self.robot.l2 or d < abs(self.robot.l1 - self.robot.l2):
            return []

        # Elbow angle
        cos_q2 = (d**2 - self.robot.l1**2 - self.robot.l2**2) / (2 * self.robot.l1 * self.robot.l2)
        cos_q2 = np.clip(cos_q2, -1, 1)

        solutions = []
        for sign in [1, -1]:
            q2 = sign * np.arccos(cos_q2)
            q1 = np.arctan2(y, x) - np.arctan2(
                self.robot.l2 * np.sin(q2), self.robot.l1 + self.robot.l2 * np.cos(q2)
            )
            solutions.append(np.array([q1, q2]))

        return solutions

    def solve_valid(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Return only valid IK solutions (within limits, collision-free).

        Note: q_init is ignored (analytical solver finds all solutions).
        """
        solutions = self.solve(pose)
        lower, upper = self.robot.joint_limits

        valid = []
        for q in solutions:
            # Check joint limits
            if not (np.all(q >= lower) and np.all(q <= upper)):
                continue
            # Check collisions
            if not self.collision.is_valid(q):
                continue
            valid.append(q)

        return valid


class TestRRTree:
    def test_init(self):
        root = np.array([0.0, 0.0])
        tree = RRTree(root)
        assert len(tree) == 1
        assert np.allclose(tree.nodes[0].config, root)

    def test_add_node(self):
        tree = RRTree(np.array([0.0, 0.0]))
        idx = tree.add_node(np.array([1.0, 1.0]), parent_idx=0)
        assert idx == 1
        assert len(tree) == 2
        assert tree.nodes[1].parent == 0

    def test_nearest(self):
        tree = RRTree(np.array([0.0, 0.0]))
        tree.add_node(np.array([1.0, 0.0]), 0)
        tree.add_node(np.array([0.0, 1.0]), 0)

        assert tree.nearest(np.array([0.9, 0.1])) == 1
        assert tree.nearest(np.array([0.1, 0.9])) == 2

    def test_path_to_root(self):
        tree = RRTree(np.array([0.0, 0.0]))
        idx1 = tree.add_node(np.array([1.0, 0.0]), 0)
        idx2 = tree.add_node(np.array([2.0, 0.0]), idx1)

        path = tree.get_path_to_root(idx2)
        assert len(path) == 3
        assert np.allclose(path[0], [0.0, 0.0])
        assert np.allclose(path[1], [1.0, 0.0])
        assert np.allclose(path[2], [2.0, 0.0])

    def test_rrtree_multiple_roots(self):
        """Test RRTree with multiple root configurations."""
        configs = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        tree = RRTree(configs)
        assert len(tree) == 2
        assert tree.nodes[0].parent is None
        assert tree.nodes[1].parent is None
        assert tree.num_roots == 2

    def test_rrtree_backward_compatibility(self):
        """Test RRTree still accepts single config."""
        tree = RRTree(np.array([0.0, 0.0]))
        assert len(tree) == 1
        assert tree.num_roots == 1

    def test_rrtree_nearest_with_multiple_roots(self):
        """Test nearest neighbor search works with multiple roots."""
        configs = [np.array([0.0, 0.0]), np.array([2.0, 2.0])]
        tree = RRTree(configs)

        # Query closer to first root
        assert tree.nearest(np.array([0.1, 0.1])) == 0
        # Query closer to second root
        assert tree.nearest(np.array([1.9, 1.9])) == 1


class TestCBiRRT:
    def test_plan_simple(self):
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik_solver=ik,
            collision_checker=collision,
        )

        start = np.array([0.0, 0.0])

        # Goal: reach position (1.5, 0.5)
        T0_w = np.eye(4)
        T0_w[0, 3] = 1.5
        T0_w[1, 3] = 0.5

        goal_tsr = TSR(
            T0_w=T0_w,
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.1, 0.1],
                [-0.1, 0.1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]),
        )

        path = planner.plan(start, goal_tsrs=[goal_tsr], seed=42)

        assert path is not None
        assert len(path) >= 2
        assert np.allclose(path[0], start)

        # Check goal reached
        final_pose = robot.forward_kinematics(path[-1])
        dist, _ = goal_tsr.distance(final_pose)
        assert dist < 0.2  # Within tolerance

    def test_start_in_collision_raises(self):
        class BlockingCollisionChecker:
            def is_valid(self, q):
                return False

            def is_valid_batch(self, qs):
                return np.zeros(len(qs), dtype=bool)

        planner = CBiRRT(
            robot=MockRobotModel(),
            ik_solver=MockIKSolver(),
            collision_checker=BlockingCollisionChecker(),
        )

        with pytest.raises(AllStartConfigurationsInCollision):
            planner.plan(np.array([0.0, 0.0]), goal_tsrs=[])

    @pytest.mark.parametrize(
        "extend_steps,connect_steps,variant_name",
        [
            (None, None, "CON-CON"),
            (5, 5, "EXT-EXT"),
            (5, None, "EXT-CON"),
            (None, 5, "CON-EXT"),
        ],
    )
    def test_all_variants(self, extend_steps, connect_steps, variant_name):
        """Test all 4 BiRRT variants: CON-CON, EXT-EXT, EXT-CON, CON-EXT."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        config = CBiRRTConfig(
            step_size=0.2,
            extend_steps=extend_steps,
            connect_steps=connect_steps,
        )
        planner = CBiRRT(
            robot=robot,
            ik_solver=ik,
            collision_checker=collision,
            config=config,
        )

        start = np.array([0.0, 0.0])

        # Goal: reach position (1.5, 0.5)
        T0_w = np.eye(4)
        T0_w[0, 3] = 1.5
        T0_w[1, 3] = 0.5

        goal_tsr = TSR(
            T0_w=T0_w,
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.1, 0.1],
                [-0.1, 0.1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]),
        )

        result = planner.plan(start, goal_tsrs=[goal_tsr], seed=42, return_details=True)

        assert result.success, f"{variant_name} failed to find path"
        assert len(result.path) >= 2
        assert np.allclose(result.path[0], start)

        # Check goal reached
        final_pose = robot.forward_kinematics(result.path[-1])
        dist, _ = goal_tsr.distance(final_pose)
        assert dist < 0.2, f"{variant_name} did not reach goal (dist={dist})"

    def test_start_tsrs(self):
        """Test planning with start sampled from TSRs."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik_solver=ik,
            collision_checker=collision,
        )

        # Start TSR: positions near (1.0, 1.0)
        T0_w_start = np.eye(4)
        T0_w_start[0, 3] = 1.0
        T0_w_start[1, 3] = 1.0

        start_tsr = TSR(
            T0_w=T0_w_start,
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.1, 0.1],
                [-0.1, 0.1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]),
        )

        # Goal TSR: positions near (1.5, 0.5)
        T0_w_goal = np.eye(4)
        T0_w_goal[0, 3] = 1.5
        T0_w_goal[1, 3] = 0.5

        goal_tsr = TSR(
            T0_w=T0_w_goal,
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.1, 0.1],
                [-0.1, 0.1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]),
        )

        # Plan with start=None, using start_tsrs
        result = planner.plan(
            start=None,
            goal_tsrs=[goal_tsr],
            start_tsrs=[start_tsr],
            seed=42,
            return_details=True,
        )

        assert result.success
        assert len(result.path) >= 2

        # Verify start is in start TSR region
        start_pose = robot.forward_kinematics(result.path[0])
        start_dist, _ = start_tsr.distance(start_pose)
        assert start_dist < 0.2

        # Verify goal is reached
        final_pose = robot.forward_kinematics(result.path[-1])
        goal_dist, _ = goal_tsr.distance(final_pose)
        assert goal_dist < 0.2

    def test_no_start_raises(self):
        """Test that providing neither start nor start_tsrs raises error."""
        planner = CBiRRT(
            robot=MockRobotModel(),
            ik_solver=MockIKSolver(),
            collision_checker=MockCollisionChecker(),
        )

        goal_tsr = TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.zeros((6, 2)),
        )

        with pytest.raises(ValueError, match="No valid start configurations available"):
            planner.plan(start=None, goal_tsrs=[goal_tsr])

    def test_plan_multiple_starts(self):
        """Test planning from multiple start configs."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik_solver=ik,
            collision_checker=collision,
        )

        starts = [np.array([0.0, 0.0]), np.array([0.1, 0.1])]

        # Goal TSR at (1.5, 0.5)
        T0_w = np.eye(4)
        T0_w[0, 3] = 1.5
        T0_w[1, 3] = 0.5
        goal_tsr = TSR(
            T0_w=T0_w,
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.1, 0.1],
                [-0.1, 0.1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]),
        )

        result = planner.plan(
            start=starts, goal_tsrs=[goal_tsr], seed=42, return_details=True
        )

        assert result.success
        # Path starts from one of the provided starts
        assert any(np.allclose(result.path[0], s) for s in starts)

    def test_plan_multiple_goals(self):
        """Test planning to multiple goal configs."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik_solver=ik,
            collision_checker=collision,
        )

        start = np.array([0.0, 0.0])
        # Two valid goal configurations
        goals = [np.array([np.pi / 2, 0.0]), np.array([np.pi / 2, np.pi / 4])]

        path = planner.plan(start=start, goal=goals, seed=42)

        assert path is not None
        # Path ends at one of the provided goals (within tolerance)
        assert any(np.allclose(path[-1], g, atol=0.05) for g in goals)

    def test_plan_multiple_to_multiple(self):
        """Test planning with multiple starts and goals."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik_solver=ik,
            collision_checker=collision,
        )

        starts = [np.array([0.0, 0.0]), np.array([0.1, 0.1])]
        goals = [np.array([np.pi / 2, 0.0]), np.array([np.pi / 2, np.pi / 4])]

        path = planner.plan(start=starts, goal=goals, seed=42)
        assert path is not None

    def test_invalid_start_in_list_raises(self):
        """Test that invalid start config in list raises clear error."""

        class AlwaysInvalidChecker:
            def is_valid(self, q):
                return False

            def is_valid_batch(self, qs):
                return np.zeros(len(qs), dtype=bool)

        robot = MockRobotModel()
        ik = MockIKSolver(robot, MockCollisionChecker())
        planner = CBiRRT(robot, ik, AlwaysInvalidChecker())

        starts = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]

        T0_w = np.eye(4)
        T0_w[0, 3] = 1.5
        T0_w[1, 3] = 0.5
        goal_tsr = TSR(
            T0_w=T0_w,
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.1, 0.1],
                [-0.1, 0.1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]),
        )

        with pytest.raises(AllStartConfigurationsInCollision):
            planner.plan(start=starts, goal_tsrs=[goal_tsr])

    def test_backward_compatibility(self):
        """Test that old API still works."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik_solver=ik,
            collision_checker=collision,
        )

        start_config = np.array([0.0, 0.0])

        T0_w = np.eye(4)
        T0_w[0, 3] = 1.5
        T0_w[1, 3] = 0.5
        goal_tsr = TSR(
            T0_w=T0_w,
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.1, 0.1],
                [-0.1, 0.1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]),
        )

        # Old style - positional arg, only goal_tsrs
        path = planner.plan(start_config, goal_tsrs=[goal_tsr], seed=42)
        assert path is not None

    def test_single_goal_config(self):
        """Test planning with single goal configuration (new API)."""
        robot = MockRobotModel()
        collision = MockCollisionChecker()
        ik = MockIKSolver(robot, collision)

        planner = CBiRRT(
            robot=robot,
            ik_solver=ik,
            collision_checker=collision,
        )

        start = np.array([0.0, 0.0])
        goal = np.array([np.pi / 2, 0.0])

        path = planner.plan(start=start, goal=goal, seed=42)

        assert path is not None
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], goal, atol=0.05)


class TestConfigValidation:
    """Tests for start/goal configuration validation."""

    def test_filters_invalid_starts_proceeds_with_valid(self, caplog):
        """Test that invalid starts are filtered but planning proceeds with valid ones."""

        class SelectiveCollisionChecker:
            """Only the first config is in collision."""

            def is_valid(self, q):
                # Config [0, 0] is in collision, others are fine
                return not (q[0] == 0.0 and q[1] == 0.0)

            def is_valid_batch(self, qs):
                return np.array([self.is_valid(q) for q in qs])

        robot = MockRobotModel()
        ik = MockIKSolver(robot, MockCollisionChecker())
        planner = CBiRRT(robot, ik, SelectiveCollisionChecker())

        # First start is in collision, second is valid
        starts = [np.array([0.0, 0.0]), np.array([0.5, 0.5])]
        goal = np.array([1.0, 0.0])

        import logging

        with caplog.at_level(logging.WARNING):
            path = planner.plan(start=starts, goal=goal, seed=42)

        # Should succeed with the valid start
        assert path is not None
        # Should have logged a warning about the filtered config
        assert "Filtered 1 invalid start" in caplog.text
        assert "Start[0]: in collision" in caplog.text

    def test_all_goals_in_collision_raises(self):
        """Test that all goals in collision raises AllGoalConfigurationsInCollision."""
        from pycbirrt import AllGoalConfigurationsInCollision

        class GoalBlockingChecker:
            """Start is valid, goals are not."""

            def __init__(self, goal_configs):
                self.goal_configs = [tuple(g) for g in goal_configs]

            def is_valid(self, q):
                return tuple(q) not in self.goal_configs

            def is_valid_batch(self, qs):
                return np.array([self.is_valid(q) for q in qs])

        robot = MockRobotModel()
        start = np.array([0.0, 0.0])
        goals = [np.array([1.0, 0.0]), np.array([1.0, 1.0])]

        ik = MockIKSolver(robot, MockCollisionChecker())
        planner = CBiRRT(robot, ik, GoalBlockingChecker(goals))

        with pytest.raises(AllGoalConfigurationsInCollision) as exc_info:
            planner.plan(start=start, goal=goals, seed=42)

        assert exc_info.value.n_configs == 2
        assert "Goal[0]: in collision" in str(exc_info.value)
        assert "Goal[1]: in collision" in str(exc_info.value)

    def test_exception_contains_details(self):
        """Test that exception message contains details about which configs failed."""

        class AlwaysInvalidChecker:
            def is_valid(self, q):
                return False

            def is_valid_batch(self, qs):
                return np.zeros(len(qs), dtype=bool)

        robot = MockRobotModel()
        ik = MockIKSolver(robot, MockCollisionChecker())
        planner = CBiRRT(robot, ik, AlwaysInvalidChecker())

        starts = [np.array([0.0, 0.0]), np.array([0.5, 0.5]), np.array([1.0, 1.0])]

        with pytest.raises(AllStartConfigurationsInCollision) as exc_info:
            planner.plan(start=starts, goal=np.array([2.0, 0.0]), seed=42)

        exc = exc_info.value
        assert exc.n_configs == 3
        assert len(exc.details) == 3
        assert "All 3 start configuration(s) in collision" in str(exc)

    def test_filters_invalid_goals_proceeds_with_valid(self, caplog):
        """Test that invalid goals are filtered but planning proceeds with valid ones."""

        class SelectiveGoalChecker:
            """Only goal [1.0, 1.0] is in collision."""

            def is_valid(self, q):
                return not (abs(q[0] - 1.0) < 0.01 and abs(q[1] - 1.0) < 0.01)

            def is_valid_batch(self, qs):
                return np.array([self.is_valid(q) for q in qs])

        robot = MockRobotModel()
        ik = MockIKSolver(robot, MockCollisionChecker())
        planner = CBiRRT(robot, ik, SelectiveGoalChecker())

        start = np.array([0.0, 0.0])
        # First goal is in collision, second is valid
        goals = [np.array([1.0, 1.0]), np.array([1.0, 0.0])]

        import logging

        with caplog.at_level(logging.WARNING):
            path = planner.plan(start=start, goal=goals, seed=42)

        # Should succeed with the valid goal
        assert path is not None
        # Should have logged a warning
        assert "Filtered 1 invalid goal" in caplog.text
