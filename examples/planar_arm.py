"""Example: Simple 2-DOF planar arm planning with CBiRRT.

This example demonstrates three planning scenarios:
1. Basic planning with fixed start and goal TSR
2. Planning with both start and goal TSRs (no fixed configurations)
3. Constrained planning with a trajectory-wide constraint TSR

This example requires only numpy and TSR - no MuJoCo or EAIK needed.

Run with:
    python examples/planar_arm.py
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from tsr import TSR

from pycbirrt import CBiRRT, CBiRRTConfig, PlanResult
from pycbirrt.tree import RRTree


class PlanarArmRobot:
    """Simple 2-DOF planar arm."""

    def __init__(self, l1: float = 1.0, l2: float = 1.0):
        self.l1 = l1
        self.l2 = l2

    @property
    def dof(self) -> int:
        return 2

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi])

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector pose (embedded in 4x4 for TSR compatibility)."""
        x = self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1])
        y = self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1])

        T = np.eye(4)
        T[0, 3] = x
        T[1, 3] = y
        return T

    def get_joint_positions(self, q: np.ndarray) -> list[np.ndarray]:
        """Get positions of base, elbow, and end-effector for visualization."""
        base = np.array([0.0, 0.0])
        elbow = np.array([
            self.l1 * np.cos(q[0]),
            self.l1 * np.sin(q[0]),
        ])
        ee = np.array([
            self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1]),
            self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1]),
        ])
        return [base, elbow, ee]


class PlanarArmIK:
    """Analytical IK for 2-DOF planar arm."""

    def __init__(
        self,
        robot: PlanarArmRobot,
        collision_checker: "CircleObstacleChecker",
    ):
        self.robot = robot
        self.collision = collision_checker

    def solve(self, pose: np.ndarray) -> list[np.ndarray]:
        """Return all kinematic IK solutions (unvalidated)."""
        x, y = pose[0, 3], pose[1, 3]
        d = np.sqrt(x**2 + y**2)

        # Check reachability
        if d > self.robot.l1 + self.robot.l2 - 1e-6 or d < abs(self.robot.l1 - self.robot.l2) + 1e-6:
            return []

        # Elbow angle via law of cosines
        cos_q2 = (d**2 - self.robot.l1**2 - self.robot.l2**2) / (2 * self.robot.l1 * self.robot.l2)
        cos_q2 = np.clip(cos_q2, -1, 1)

        solutions = []
        for sign in [1, -1]:  # Elbow up / elbow down
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


class CircleObstacleChecker:
    """Collision checker with circular obstacles."""

    def __init__(self, robot: PlanarArmRobot, obstacles: list[tuple[np.ndarray, float]]):
        """
        Args:
            robot: The planar arm robot
            obstacles: List of (center, radius) tuples
        """
        self.robot = robot
        self.obstacles = obstacles

    def is_valid(self, q: np.ndarray) -> bool:
        """Check if arm configuration collides with any obstacle."""
        positions = self.robot.get_joint_positions(q)

        # Check each link (base->elbow, elbow->ee)
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i + 1]

            for center, radius in self.obstacles:
                if self._segment_circle_collision(p1, p2, center, radius):
                    return False

        return True

    def _segment_circle_collision(
        self, p1: np.ndarray, p2: np.ndarray, center: np.ndarray, radius: float
    ) -> bool:
        """Check if line segment intersects circle."""
        d = p2 - p1
        f = p1 - center

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return False

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        # Check if intersection is within segment
        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)

    def is_valid_batch(self, qs: np.ndarray) -> np.ndarray:
        return np.array([self.is_valid(q) for q in qs])


def visualize_result(
    robot: PlanarArmRobot,
    path: list[np.ndarray],
    obstacles: list[tuple[np.ndarray, float]],
    tree_start: RRTree,
    tree_goal: RRTree,
    collision_checker: CircleObstacleChecker,
    start_region: tuple[np.ndarray, float] | None = None,
    goal_region: tuple[np.ndarray, float] | None = None,
    constraint_region: tuple[float, float, float, float] | None = None,
    title: str = "CBiRRT Planning Result",
    filename: str = "planar_arm_result.png",
):
    """Visualize planning result with workspace path and C-space trees side by side.

    Args:
        robot: The planar arm robot
        path: List of joint configurations
        obstacles: List of (center, radius) obstacle tuples
        tree_start: RRT tree rooted at start
        tree_goal: RRT tree rooted at goal
        collision_checker: Collision checker for C-space visualization
        start_region: Optional (center, radius) for start TSR visualization
        goal_region: Optional (center, radius) for goal TSR visualization
        constraint_region: Optional (x_min, x_max, y_min, y_max) for constraint TSR
        title: Plot title
        filename: Output filename
    """
    fig, (ax_ws, ax_cs) = plt.subplots(1, 2, figsize=(16, 7))

    # === Left panel: Workspace visualization ===
    # Draw constraint region first (behind everything)
    if constraint_region is not None:
        x_min, x_max, y_min, y_max = constraint_region
        rect = Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            color="yellow", alpha=0.2, label="Constraint region"
        )
        ax_ws.add_patch(rect)

    # Draw obstacles
    for center, radius in obstacles:
        circle = Circle(center, radius, color="red", alpha=0.5)
        ax_ws.add_patch(circle)

    # Draw start region
    if start_region is not None:
        center, radius = start_region
        circle = Circle(center, radius, color="blue", alpha=0.2, label="Start region")
        ax_ws.add_patch(circle)

    # Draw goal region
    if goal_region is not None:
        center, radius = goal_region
        circle = Circle(center, radius, color="green", alpha=0.2, label="Goal region")
        ax_ws.add_patch(circle)

    # Draw path (fading from blue to green)
    n_frames = len(path)
    for i, q in enumerate(path):
        alpha = 0.2 + 0.8 * (i / n_frames)
        positions = robot.get_joint_positions(q)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        color = plt.cm.viridis(i / n_frames)
        ax_ws.plot(xs, ys, "o-", color=color, alpha=alpha, linewidth=2, markersize=5)

    # Draw start and end prominently
    start_positions = robot.get_joint_positions(path[0])
    end_positions = robot.get_joint_positions(path[-1])

    ax_ws.plot(
        [p[0] for p in start_positions],
        [p[1] for p in start_positions],
        "bo-",
        linewidth=4,
        markersize=10,
        label="Start config",
    )
    ax_ws.plot(
        [p[0] for p in end_positions],
        [p[1] for p in end_positions],
        "go-",
        linewidth=4,
        markersize=10,
        label="End config",
    )

    ax_ws.set_xlim(-2.5, 2.5)
    ax_ws.set_ylim(-2.5, 2.5)
    ax_ws.set_aspect("equal")
    ax_ws.grid(True, alpha=0.3)
    ax_ws.legend(loc="upper left")
    ax_ws.set_title(f"Workspace ({len(path)} waypoints)")
    ax_ws.set_xlabel("x")
    ax_ws.set_ylabel("y")

    # === Right panel: Configuration space visualization ===
    # Draw collision regions in C-space by sampling
    q1_range = np.linspace(-np.pi, np.pi, 100)
    q2_range = np.linspace(-np.pi, np.pi, 100)
    Q1, Q2 = np.meshgrid(q1_range, q2_range)
    collision_map = np.zeros_like(Q1)

    for i in range(Q1.shape[0]):
        for j in range(Q1.shape[1]):
            q = np.array([Q1[i, j], Q2[i, j]])
            collision_map[i, j] = 0 if collision_checker.is_valid(q) else 1

    ax_cs.contourf(Q1, Q2, collision_map, levels=[0.5, 1.5], colors=["red"], alpha=0.3)

    # Draw tree edges
    def draw_tree(tree: RRTree, color: str, label: str):
        for node in tree.nodes:
            if node.parent is not None:
                parent = tree.nodes[node.parent]
                ax_cs.plot(
                    [parent.config[0], node.config[0]],
                    [parent.config[1], node.config[1]],
                    color=color,
                    alpha=0.4,
                    linewidth=0.5,
                )
        # Draw nodes
        configs = np.array([n.config for n in tree.nodes])
        ax_cs.scatter(configs[:, 0], configs[:, 1], c=color, s=5, alpha=0.6, label=label)

    draw_tree(tree_start, "blue", f"Start tree ({len(tree_start)} nodes)")
    draw_tree(tree_goal, "green", f"Goal tree ({len(tree_goal)} nodes)")

    # Draw path
    path_arr = np.array(path)
    ax_cs.plot(path_arr[:, 0], path_arr[:, 1], "k-", linewidth=2, label="Path")
    ax_cs.scatter(path_arr[:, 0], path_arr[:, 1], c="black", s=30, zorder=5)

    # Mark start and goal
    ax_cs.scatter([tree_start.nodes[0].config[0]], [tree_start.nodes[0].config[1]],
               c="blue", s=200, marker="*", edgecolors="black", linewidths=2, zorder=10, label="Start")
    ax_cs.scatter([tree_goal.nodes[0].config[0]], [tree_goal.nodes[0].config[1]],
               c="green", s=200, marker="*", edgecolors="black", linewidths=2, zorder=10, label="Goal")

    ax_cs.set_xlim(-np.pi, np.pi)
    ax_cs.set_ylim(-np.pi, np.pi)
    ax_cs.set_xlabel("q1 (rad)")
    ax_cs.set_ylabel("q2 (rad)")
    ax_cs.set_aspect("equal")
    ax_cs.grid(True, alpha=0.3)
    ax_cs.legend(loc="upper right")
    ax_cs.set_title("Configuration Space")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved visualization to {filename}")
    plt.show()


def make_position_tsr(x: float, y: float, tolerance: float = 0.1) -> TSR:
    """Create a TSR for a position with given tolerance."""
    T0_w = np.eye(4)
    T0_w[0, 3] = x
    T0_w[1, 3] = y

    return TSR(
        T0_w=T0_w,
        Tw_e=np.eye(4),
        Bw=np.array([
            [-tolerance, tolerance],  # x tolerance
            [-tolerance, tolerance],  # y tolerance
            [0, 0],                   # z (unused in 2D)
            [0, 0],                   # roll
            [0, 0],                   # pitch
            [0, 0],                   # yaw
        ]),
    )


def make_y_constraint_tsr(y_min: float, y_max: float) -> TSR:
    """Create a TSR that constrains the end-effector y-coordinate.

    This creates a "horizontal band" constraint where the end-effector
    must stay within y_min <= y <= y_max.
    """
    # Center the TSR at y=(y_min+y_max)/2
    T0_w = np.eye(4)
    T0_w[1, 3] = (y_min + y_max) / 2

    y_range = (y_max - y_min) / 2

    return TSR(
        T0_w=T0_w,
        Tw_e=np.eye(4),
        Bw=np.array([
            [-10, 10],              # x: effectively unconstrained
            [-y_range, y_range],    # y: constrained to band
            [0, 0],                 # z
            [0, 0],                 # roll
            [0, 0],                 # pitch
            [0, 0],                 # yaw
        ]),
    )


def example_basic():
    """Example 1: Basic planning with fixed start and goal TSR."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Planning (fixed start, goal TSR)")
    print("=" * 60)

    robot = PlanarArmRobot(l1=1.0, l2=1.0)
    obstacles = [
        (np.array([1.2, 0.5]), 0.2),
        (np.array([0.5, 1.2]), 0.15),
    ]
    collision_checker = CircleObstacleChecker(robot, obstacles)
    ik = PlanarArmIK(robot, collision_checker)

    config = CBiRRTConfig(
        step_size=0.3,
        goal_bias=0.15,
        smooth_path=True,
        smoothing_iterations=50,
        angular_joints=(True, True),  # Both joints are rotational
    )
    planner = CBiRRT(
        robot=robot,
        ik_solver=ik,
        collision_checker=collision_checker,
        config=config,
    )

    # Fixed start configuration
    start = np.array([0.0, 0.0])

    # Goal TSR at (1.5, 1.0) with tolerance
    goal_pos = np.array([1.5, 1.0])
    goal_tsr = make_position_tsr(goal_pos[0], goal_pos[1], tolerance=0.1)

    print(f"  Start: {start} (fixed)")
    print(f"  Goal region: ({goal_pos[0]}, {goal_pos[1]}) +/- 0.1")
    print(f"  Obstacles: {len(obstacles)}")

    result = planner.plan(start, goal_tsrs=[goal_tsr], seed=42, return_details=True)

    print(f"  Iterations: {result.iterations}")
    print(f"  Start tree nodes: {len(result.tree_start)}")
    print(f"  Goal tree nodes: {len(result.tree_goal)}")

    if not result.success:
        print("No path found!")
        return

    print(f"Found path with {len(result.path)} waypoints")

    final_pose = robot.forward_kinematics(result.path[-1])
    final_pos = final_pose[:2, 3]
    print(f"  Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")

    visualize_result(
        robot, result.path, obstacles,
        result.tree_start, result.tree_goal, collision_checker,
        goal_region=(goal_pos, 0.1),
        title="Example 1: Basic Planning",
        filename="example1_result.png",
    )


def example_start_goal_tsrs():
    """Example 2: Planning with both start and goal TSRs."""
    print("\n" + "=" * 60)
    print("Example 2: Planning with Start and Goal TSRs")
    print("=" * 60)

    robot = PlanarArmRobot(l1=1.0, l2=1.0)
    # Small obstacle that doesn't block too much
    obstacles = [
        (np.array([1.0, 0.0]), 0.15),
    ]
    collision_checker = CircleObstacleChecker(robot, obstacles)
    ik = PlanarArmIK(robot, collision_checker)

    config = CBiRRTConfig(
        step_size=0.3,
        goal_bias=0.15,
        tsr_tolerance=0.05,  # Larger tolerance for tree connection
        smooth_path=True,
        smoothing_iterations=50,
        angular_joints=(True, True),  # Both joints are rotational
    )
    planner = CBiRRT(
        robot=robot,
        ik_solver=ik,
        collision_checker=collision_checker,
        config=config,
    )

    # Start TSR: anywhere near (1.5, 0.5) - upper right quadrant
    start_pos = np.array([1.5, 0.5])
    start_tsr = make_position_tsr(start_pos[0], start_pos[1], tolerance=0.2)

    # Goal TSR: anywhere near (0.5, 1.5) - upper left quadrant
    goal_pos = np.array([0.5, 1.5])
    goal_tsr = make_position_tsr(goal_pos[0], goal_pos[1], tolerance=0.2)

    print(f"  Start region: ({start_pos[0]}, {start_pos[1]}) +/- 0.2")
    print(f"  Goal region: ({goal_pos[0]}, {goal_pos[1]}) +/- 0.2")
    print(f"  Obstacles: {len(obstacles)}")

    result = planner.plan(
        start=None,  # No fixed start - sample from start_tsrs
        goal_tsrs=[goal_tsr],
        start_tsrs=[start_tsr],
        seed=42,
        return_details=True,
    )

    print(f"  Iterations: {result.iterations}")
    print(f"  Start tree nodes: {len(result.tree_start)}")
    print(f"  Goal tree nodes: {len(result.tree_goal)}")

    if not result.success:
        print("No path found!")
        return

    print(f"Found path with {len(result.path)} waypoints")

    # Verify start and goal
    start_pose = robot.forward_kinematics(result.path[0])
    final_pose = robot.forward_kinematics(result.path[-1])
    print(f"  Sampled start: ({start_pose[0, 3]:.3f}, {start_pose[1, 3]:.3f})")
    print(f"  Final position: ({final_pose[0, 3]:.3f}, {final_pose[1, 3]:.3f})")

    visualize_result(
        robot, result.path, obstacles,
        result.tree_start, result.tree_goal, collision_checker,
        start_region=(start_pos, 0.2),
        goal_region=(goal_pos, 0.2),
        title="Example 2: Start and Goal TSRs",
        filename="example2_result.png",
    )


def example_constrained():
    """Example 3: Constrained planning with trajectory-wide constraint TSR.

    The end-effector must stay within a horizontal band (y constraint)
    throughout the entire trajectory.
    """
    print("\n" + "=" * 60)
    print("Example 3: Constrained Planning (y-band constraint)")
    print("=" * 60)

    robot = PlanarArmRobot(l1=1.0, l2=1.0)
    # No obstacles for this example - the constraint is the challenge
    obstacles = []
    collision_checker = CircleObstacleChecker(robot, obstacles)
    ik = PlanarArmIK(robot, collision_checker)

    config = CBiRRTConfig(
        step_size=0.2,
        goal_bias=0.1,
        smooth_path=True,
        smoothing_iterations=50,
        tsr_tolerance=0.05,
        angular_joints=(True, True),  # Both joints are rotational
    )
    planner = CBiRRT(
        robot=robot,
        ik_solver=ik,
        collision_checker=collision_checker,
        config=config,
    )

    # Constraint: end-effector must stay in y-band [0.8, 1.2]
    y_min, y_max = 0.8, 1.2
    constraint_tsr = make_y_constraint_tsr(y_min, y_max)

    # Start: left side of the band
    start_pos = np.array([-1.5, 1.0])
    start_tsr = make_position_tsr(start_pos[0], start_pos[1], tolerance=0.1)

    # Goal: right side of the band
    goal_pos = np.array([1.5, 1.0])
    goal_tsr = make_position_tsr(goal_pos[0], goal_pos[1], tolerance=0.1)

    print(f"  Start region: ({start_pos[0]}, {start_pos[1]}) +/- 0.1")
    print(f"  Goal region: ({goal_pos[0]}, {goal_pos[1]}) +/- 0.1")
    print(f"  Constraint: y in [{y_min}, {y_max}] (horizontal band)")

    result = planner.plan(
        start=None,
        goal_tsrs=[goal_tsr],
        start_tsrs=[start_tsr],
        constraint_tsrs=[constraint_tsr],
        seed=42,
        return_details=True,
    )

    print(f"  Iterations: {result.iterations}")
    print(f"  Start tree nodes: {len(result.tree_start)}")
    print(f"  Goal tree nodes: {len(result.tree_goal)}")

    if not result.success:
        print("No path found!")
        print("  (Constrained planning is harder - the path must stay on the constraint manifold)")
        return

    print(f"Found path with {len(result.path)} waypoints")

    # Verify constraint satisfaction along path
    max_violation = 0.0
    for q in result.path:
        pose = robot.forward_kinematics(q)
        y = pose[1, 3]
        if y < y_min:
            max_violation = max(max_violation, y_min - y)
        elif y > y_max:
            max_violation = max(max_violation, y - y_max)

    print(f"  Max constraint violation: {max_violation:.4f}")

    start_pose = robot.forward_kinematics(result.path[0])
    final_pose = robot.forward_kinematics(result.path[-1])
    print(f"  Start position: ({start_pose[0, 3]:.3f}, {start_pose[1, 3]:.3f})")
    print(f"  Final position: ({final_pose[0, 3]:.3f}, {final_pose[1, 3]:.3f})")

    visualize_result(
        robot, result.path, obstacles,
        result.tree_start, result.tree_goal, collision_checker,
        start_region=(start_pos, 0.1),
        goal_region=(goal_pos, 0.1),
        constraint_region=(-2.5, 2.5, y_min, y_max),
        title="Example 3: Constrained Planning (y-band)",
        filename="example3_result.png",
    )


def main():
    parser = argparse.ArgumentParser(description="CBiRRT planar arm examples")
    parser.add_argument(
        "--example", "-e",
        type=int,
        choices=[1, 2, 3],
        help="Run specific example (1=basic, 2=start/goal TSRs, 3=constrained)",
    )
    args = parser.parse_args()

    if args.example == 1:
        example_basic()
    elif args.example == 2:
        example_start_goal_tsrs()
    elif args.example == 3:
        example_constrained()
    else:
        # Run all examples
        example_basic()
        example_start_goal_tsrs()
        example_constrained()


if __name__ == "__main__":
    main()
