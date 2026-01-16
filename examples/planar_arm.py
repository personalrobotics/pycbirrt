"""Example: Simple 2-DOF planar arm planning with CBiRRT.

This example requires only numpy and TSR - no MuJoCo or EAIK needed.

Run with:
    python examples/planar_arm.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tsr import TSR

from pycbirrt import CBiRRT, CBiRRTConfig


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

    def __init__(self, l1: float = 1.0, l2: float = 1.0):
        self.l1 = l1
        self.l2 = l2

    def solve(self, pose: np.ndarray) -> list[np.ndarray]:
        x, y = pose[0, 3], pose[1, 3]
        d = np.sqrt(x**2 + y**2)

        # Check reachability
        if d > self.l1 + self.l2 - 1e-6 or d < abs(self.l1 - self.l2) + 1e-6:
            return []

        # Elbow angle via law of cosines
        cos_q2 = (d**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_q2 = np.clip(cos_q2, -1, 1)

        solutions = []
        for sign in [1, -1]:  # Elbow up / elbow down
            q2 = sign * np.arccos(cos_q2)
            q1 = np.arctan2(y, x) - np.arctan2(
                self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2)
            )
            solutions.append(np.array([q1, q2]))

        return solutions

    def solve_batch(self, poses: np.ndarray) -> list[list[np.ndarray]]:
        return [self.solve(p) for p in poses]


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


def visualize_path(
    robot: PlanarArmRobot,
    path: list[np.ndarray],
    obstacles: list[tuple[np.ndarray, float]],
    goal_pos: np.ndarray,
):
    """Visualize the planned path."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw obstacles
    for center, radius in obstacles:
        circle = Circle(center, radius, color="red", alpha=0.5)
        ax.add_patch(circle)

    # Draw goal region
    goal_circle = Circle(goal_pos[:2], 0.1, color="green", alpha=0.3, label="Goal")
    ax.add_patch(goal_circle)

    # Draw path (fading from blue to green)
    n_frames = len(path)
    for i, q in enumerate(path):
        alpha = 0.2 + 0.8 * (i / n_frames)
        positions = robot.get_joint_positions(q)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        color = plt.cm.viridis(i / n_frames)
        ax.plot(xs, ys, "o-", color=color, alpha=alpha, linewidth=2, markersize=5)

    # Draw start and end prominently
    start_positions = robot.get_joint_positions(path[0])
    end_positions = robot.get_joint_positions(path[-1])

    ax.plot(
        [p[0] for p in start_positions],
        [p[1] for p in start_positions],
        "bo-",
        linewidth=4,
        markersize=10,
        label="Start",
    )
    ax.plot(
        [p[0] for p in end_positions],
        [p[1] for p in end_positions],
        "go-",
        linewidth=4,
        markersize=10,
        label="End",
    )

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"CBiRRT Path ({len(path)} waypoints)")

    plt.savefig("planar_arm_path.png", dpi=150)
    print("Saved visualization to planar_arm_path.png")
    plt.show()


def main():
    # Create robot
    robot = PlanarArmRobot(l1=1.0, l2=1.0)
    ik = PlanarArmIK(l1=1.0, l2=1.0)

    # Create obstacles (positioned to avoid start config but require planning)
    obstacles = [
        (np.array([1.2, 0.5]), 0.2),   # Obstacle forcing path around
        (np.array([0.5, 1.2]), 0.15),  # Upper obstacle
    ]
    collision_checker = CircleObstacleChecker(robot, obstacles)

    # Create planner
    config = CBiRRTConfig(
        max_iterations=2000,
        step_size=0.3,
        goal_bias=0.15,
        smooth_path=True,
        smoothing_iterations=50,
    )
    planner = CBiRRT(
        robot=robot,
        ik_solver=ik,
        collision_checker=collision_checker,
        config=config,
    )

    # Start configuration
    start = np.array([0.0, 0.0])  # Arm pointing right

    # Goal: reach position (1.5, 1.0) with some tolerance
    goal_pos = np.array([1.5, 1.0])
    T0_w = np.eye(4)
    T0_w[0, 3] = goal_pos[0]
    T0_w[1, 3] = goal_pos[1]

    goal_tsr = TSR(
        T0_w=T0_w,
        Tw_e=np.eye(4),
        Bw=np.array([
            [-0.1, 0.1],  # x tolerance
            [-0.1, 0.1],  # y tolerance
            [0, 0],       # z (unused in 2D)
            [0, 0],       # roll
            [0, 0],       # pitch
            [0, 0],       # yaw
        ]),
    )

    print("Planning path...")
    print(f"  Start: {start}")
    print(f"  Goal region: ({goal_pos[0]}, {goal_pos[1]}) ± 0.1")
    print(f"  Obstacles: {len(obstacles)}")

    path = planner.plan(start, goal_tsrs=[goal_tsr], seed=42)

    if path is None:
        print("No path found!")
        return

    print(f"Found path with {len(path)} waypoints")

    # Verify goal reached
    final_pose = robot.forward_kinematics(path[-1])
    final_pos = final_pose[:2, 3]
    print(f"  Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")
    print(f"  Distance to goal: {np.linalg.norm(final_pos - goal_pos):.4f}")

    # Visualize
    visualize_path(robot, path, obstacles, goal_pos)


if __name__ == "__main__":
    main()
