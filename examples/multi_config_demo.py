"""Demo of multiple start and goal configurations.

This example demonstrates the new capability to provide multiple discrete
configurations as lists. The planner grows trees from all configurations
simultaneously.
"""

import numpy as np
from tsr import TSR

# Import from parent examples
import sys
sys.path.insert(0, '.')
from planar_arm import PlanarArmRobot, PlanarArmIK, CircleObstacleChecker

from pycbirrt import CBiRRT, CBiRRTConfig


def main():
    print("\n" + "=" * 60)
    print("Multiple Start and Goal Configurations Demo")
    print("=" * 60)

    # Setup robot
    robot = PlanarArmRobot(l1=1.0, l2=1.0)
    # Small obstacle that doesn't block the start configs
    obstacles = [(np.array([0.5, 1.2]), 0.15)]
    collision_checker = CircleObstacleChecker(robot, obstacles)
    ik = PlanarArmIK(robot, collision_checker)

    config = CBiRRTConfig(
        step_size=0.3,
        smooth_path=True,
        angular_joints=(True, True),
    )
    planner = CBiRRT(
        robot=robot,
        ik_solver=ik,
        collision_checker=collision_checker,
        config=config,
    )

    # Define multiple start configurations (collision-free)
    start_configs = [
        np.array([np.pi / 4, 0.0]),    # Start 1: 45 degrees up-right
        np.array([np.pi / 4, 0.2]),    # Start 2: slightly different
    ]

    # Define multiple goal configurations
    goal_configs = [
        np.array([np.pi / 2, 0.0]),       # Goal 1: pointing up
        np.array([np.pi / 2, np.pi / 4]), # Goal 2: pointing up-left
    ]

    print(f"\nStart configs: {len(start_configs)}")
    for i, q in enumerate(start_configs):
        pose = robot.forward_kinematics(q)
        print(f"  Start {i+1}: joints={q}, ee=({pose[0,3]:.3f}, {pose[1,3]:.3f})")

    print(f"\nGoal configs: {len(goal_configs)}")
    for i, q in enumerate(goal_configs):
        pose = robot.forward_kinematics(q)
        print(f"  Goal {i+1}: joints={q}, ee=({pose[0,3]:.3f}, {pose[1,3]:.3f})")

    # Plan with multiple starts and goals
    print("\nPlanning with multiple starts and goals...")
    result = planner.plan(
        start=start_configs,
        goal=goal_configs,
        seed=42,
        return_details=True,
    )

    if not result.success:
        print("Planning failed!")
        return

    print(f"\n✓ Success! Found path in {result.iterations} iterations")
    print(f"  Path length: {len(result.path)} waypoints")
    print(f"  Start tree: {len(result.tree_start)} nodes ({result.tree_start.num_roots} roots)")
    print(f"  Goal tree: {len(result.tree_goal)} nodes ({result.tree_goal.num_roots} roots)")

    # Check which start/goal were connected
    actual_start = result.path[0]
    actual_goal = result.path[-1]

    start_idx = next(i for i, s in enumerate(start_configs) if np.allclose(s, actual_start, atol=0.01))
    goal_idx = next(i for i, g in enumerate(goal_configs) if np.allclose(g, actual_goal, atol=0.1))

    print(f"\n  Connected: Start {start_idx+1} → Goal {goal_idx+1}")

    start_pose = robot.forward_kinematics(actual_start)
    goal_pose = robot.forward_kinematics(actual_goal)
    print(f"  Start ee: ({start_pose[0,3]:.3f}, {start_pose[1,3]:.3f})")
    print(f"  Goal ee: ({goal_pose[0,3]:.3f}, {goal_pose[1,3]:.3f})")


if __name__ == "__main__":
    main()
