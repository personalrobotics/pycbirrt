"""Tests for CBiRRT planner."""

import numpy as np
import pytest
from tsr import TSR

from pycbirrt import CBiRRT, CBiRRTConfig
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

    def solve(self, pose: np.ndarray) -> list[np.ndarray]:
        """Return all kinematic IK solutions (unvalidated)."""
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

    def solve_valid(self, pose: np.ndarray) -> list[np.ndarray]:
        """Return only valid IK solutions (within limits, collision-free)."""
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

        with pytest.raises(ValueError, match="Start configuration is in collision"):
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

        with pytest.raises(ValueError, match="Either start or start_tsrs must be provided"):
            planner.plan(start=None, goal_tsrs=[goal_tsr])
