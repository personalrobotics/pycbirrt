"""Demo: TSR Union planning with multiple goal locations.

This example demonstrates planning to a union of goal TSRs - the planner
finds a path to ANY goal in the set. We create two targets on opposite
sides of the robot (left and right), making it roughly equally likely
for the planner to reach either one depending on the random seed.

The video loops between start and goal, showing which target the planner chose.

Prerequisites:
    export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie
    uv pip install pycbirrt[mujoco] mediapy

Usage:
    python examples/tsr_union_demo.py                    # Render to video
    python examples/tsr_union_demo.py --loops 5          # More loops
    python examples/tsr_union_demo.py --output my.mp4    # Custom output
    mjpython examples/tsr_union_demo.py --interactive    # Interactive viewer
"""

import argparse
import os
from pathlib import Path

import numpy as np
import mujoco

from pycbirrt import CBiRRT, CBiRRTConfig
from pycbirrt.backends.mujoco import (
    MuJoCoRobotModel,
    MuJoCoCollisionChecker,
    MuJoCoIKSolver,
)
from tsr import TSR

# Optional EAIK for faster planning
try:
    from pycbirrt.backends.eaik import EAIKSolver

    EAIK_AVAILABLE = True
except ImportError:
    EAIK_AVAILABLE = False


def get_menagerie_path() -> Path:
    """Get path to MuJoCo Menagerie."""
    path = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if path is None:
        raise EnvironmentError(
            "Set MUJOCO_MENAGERIE_PATH to your mujoco_menagerie clone"
        )
    return Path(path)


def create_scene(menagerie_path: Path, target_left: np.ndarray, target_right: np.ndarray) -> "mujoco.MjModel":
    """Create scene with UR5e and two target objects."""
    ur5e_path = menagerie_path / "universal_robots_ur5e" / "ur5e.xml"
    spec = mujoco.MjSpec.from_file(str(ur5e_path))

    # High-res offscreen rendering
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    world = spec.worldbody

    # Clean white floor
    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.pos = [0, 0, -0.01]
    floor.size = [0, 0, 0.05]
    floor.rgba = [0.95, 0.95, 0.95, 1]

    # Robot base marker
    base = world.add_geom()
    base.name = "base_marker"
    base.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    base.pos = [0, 0, 0.01]
    base.size = [0.1, 0.01, 0]
    base.rgba = [0.3, 0.5, 0.7, 0.8]
    base.contype = 0
    base.conaffinity = 0

    # Table (wide enough for both targets)
    table = world.add_body()
    table.name = "table"
    table.pos = [0.45, 0, 0.30]

    table_top = table.add_geom()
    table_top.name = "table_top"
    table_top.type = mujoco.mjtGeom.mjGEOM_BOX
    table_top.size = [0.25, 0.5, 0.02]
    table_top.rgba = [0.45, 0.4, 0.35, 1]

    # Table legs
    for i, (x, y) in enumerate([(0.2, 0.45), (0.2, -0.45), (-0.2, 0.45), (-0.2, -0.45)]):
        leg = table.add_geom()
        leg.name = f"leg_{i}"
        leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        leg.pos = [x, y, -0.14]
        leg.size = [0.02, 0.12, 0]
        leg.rgba = [0.35, 0.3, 0.25, 1]
        leg.contype = 0
        leg.conaffinity = 0

    # Left target (blue)
    left = world.add_body()
    left.name = "target_left"
    left.pos = list(target_left)

    left_geom = left.add_geom()
    left_geom.name = "target_left_geom"
    left_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    left_geom.size = [0.025, 0.05, 0]
    left_geom.rgba = [0.2, 0.4, 0.9, 1]  # Blue

    # Right target (red)
    right = world.add_body()
    right.name = "target_right"
    right.pos = list(target_right)

    right_geom = right.add_geom()
    right_geom.name = "target_right_geom"
    right_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    right_geom.size = [0.025, 0.05, 0]
    right_geom.rgba = [0.9, 0.2, 0.2, 1]  # Red

    return spec.compile()


def create_goal_tsrs(target_left: np.ndarray, target_right: np.ndarray) -> tuple[TSR, TSR]:
    """Create two goal TSRs for top-down grasps at each target location.

    Both are top-down grasps (gripper pointing down), but at different
    positions on opposite sides of the robot.

    Args:
        target_left: Position of left target (negative Y)
        target_right: Position of right target (positive Y)

    Returns:
        (left_tsr, right_tsr)
    """
    standoff = 0.12  # Height above target

    # Gripper rotation for top-down: z points DOWN
    R_top_down = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    # Left target TSR
    T_left = np.eye(4)
    T_left[:3, 3] = target_left + np.array([0, 0, standoff])
    T_left[:3, :3] = R_top_down

    left_tsr = TSR(
        T0_w=T_left,
        Tw_e=np.eye(4),
        Bw=np.array([
            [-0.02, 0.02],   # x
            [-0.02, 0.02],   # y
            [0, 0.03],       # z (can be slightly higher)
            [0, 0],          # roll
            [0, 0],          # pitch
            [-np.pi, np.pi]  # yaw (any rotation around vertical)
        ])
    )

    # Right target TSR
    T_right = np.eye(4)
    T_right[:3, 3] = target_right + np.array([0, 0, standoff])
    T_right[:3, :3] = R_top_down

    right_tsr = TSR(
        T0_w=T_right,
        Tw_e=np.eye(4),
        Bw=np.array([
            [-0.02, 0.02],
            [-0.02, 0.02],
            [0, 0.03],
            [0, 0],
            [0, 0],
            [-np.pi, np.pi]
        ])
    )

    return left_tsr, right_tsr


def identify_reached_goal(
    ee_pose: np.ndarray, left_tsr: TSR, right_tsr: TSR
) -> str:
    """Identify which goal TSR was reached."""
    left_dist, _ = left_tsr.distance(ee_pose)
    right_dist, _ = right_tsr.distance(ee_pose)

    if left_dist < right_dist:
        return "LEFT (blue)"
    else:
        return "RIGHT (red)"


def render_loop_video(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    path_forward: list[np.ndarray],
    joint_names: list[str],
    num_loops: int,
    output_path: str,
    goal_label: str,
    fps: int = 60,
    width: int = 1280,
    height: int = 720,
):
    """Render a looping video: start -> goal -> start -> goal -> ..."""
    import mediapy as media

    renderer = mujoco.Renderer(model, width=width, height=height)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.35, 0.0, 0.30]
    camera.distance = 1.4
    camera.azimuth = 160  # More front view to see both sides
    camera.elevation = -25

    scene_option = mujoco.MjvOption()

    frames = []
    steps_per_waypoint = 2  # Fast movement

    # Create reverse path
    path_backward = list(reversed(path_forward))

    for loop in range(num_loops):
        # Alternate direction
        path = path_forward if loop % 2 == 0 else path_backward

        for i in range(len(path) - 1):
            for t in np.linspace(0, 1, steps_per_waypoint, endpoint=False):
                q = (1 - t) * path[i] + t * path[i + 1]

                for j, jnt in enumerate(joint_names):
                    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt)
                    data.qpos[model.jnt_qposadr[jnt_id]] = q[j]

                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                frames.append(renderer.render())

        # Hold at endpoint
        for _ in range(fps // 3):
            frames.append(renderer.render())

    media.write_video(output_path, frames, fps=fps)
    print(f"Saved looping video to {output_path}")
    print(f"  {num_loops} loops, {len(frames)} frames, ~{len(frames)/fps:.1f}s")
    print(f"  Planner chose: {goal_label}")


def visualize_interactive(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    path_forward: list[np.ndarray],
    joint_names: list[str],
    goal_label: str,
):
    """Interactive viewer with looping playback."""
    import mujoco.viewer

    print(f"Opening viewer (planner chose {goal_label})")
    print("Press ESC to close")

    path_backward = list(reversed(path_forward))
    paths = [path_forward, path_backward]

    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        print(f"Viewer not available: {e}")
        print("Use --output video.mp4 to render instead")
        return

    with viewer:
        path_idx = 0
        waypoint_idx = 0
        steps_per_waypoint = 10

        while viewer.is_running():
            path = paths[path_idx]

            if waypoint_idx < len(path) - 1:
                t = (data.time * 20) % steps_per_waypoint / steps_per_waypoint
                q = (1 - t) * path[waypoint_idx] + t * path[waypoint_idx + 1]

                if t > 0.95:
                    waypoint_idx += 1
            else:
                # Switch direction
                q = path[-1]
                if data.time % 0.5 < 0.01:  # Brief pause
                    path_idx = 1 - path_idx
                    waypoint_idx = 0

            for i, jnt in enumerate(joint_names):
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt)
                data.qpos[model.jnt_qposadr[jnt_id]] = q[i]

            mujoco.mj_forward(model, data)
            viewer.sync()
            data.time += 0.01


def main():
    parser = argparse.ArgumentParser(description="TSR Union demo with looping video")
    parser.add_argument("--output", "-o", default="/tmp/tsr_union_demo.mp4",
                        help="Output video path")
    parser.add_argument("--loops", type=int, default=4,
                        help="Number of back-and-forth loops")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (None = random each run)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Use interactive viewer instead of video")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of planning runs (to see different goals)")
    args = parser.parse_args()

    menagerie = get_menagerie_path()

    # Target positions: symmetric on left (-Y) and right (+Y)
    target_left = np.array([0.45, -0.25, 0.37])   # On table, left side
    target_right = np.array([0.45, 0.25, 0.37])   # On table, right side

    model = create_scene(menagerie, target_left, target_right)
    data = mujoco.MjData(model)

    joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # Create backends
    robot = MuJoCoRobotModel(model, data, "attachment_site", joints)
    collision = MuJoCoCollisionChecker(model, data, joints)

    if EAIK_AVAILABLE:
        print("Using EAIK analytical IK solver")
        ik_solver = EAIKSolver.for_ur5e(robot.joint_limits, collision)
        config = CBiRRTConfig(
            timeout=30.0,
            goal_bias=0.15,
            ik_num_seeds=1,
        )
    else:
        print("Using MuJoCo differential IK solver")
        ik_solver = MuJoCoIKSolver(model, data, "attachment_site", joints, collision)
        config = CBiRRTConfig(
            timeout=30.0,
            goal_bias=0.15,
            ik_num_seeds=10,
        )

    planner = CBiRRT(robot, ik_solver, collision, config)

    # Start configuration (home)
    start = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])

    # Create goal TSRs
    left_tsr, right_tsr = create_goal_tsrs(target_left, target_right)

    print("\n" + "=" * 60)
    print("TSR UNION PLANNING DEMO")
    print("=" * 60)
    print(f"Two targets on opposite sides of the robot:")
    print(f"  LEFT (blue):  {target_left}")
    print(f"  RIGHT (red):  {target_right}")
    print(f"\nPlanner will find a path to ANY target in the union.")
    print(f"Different random seeds may lead to different targets!")
    print("=" * 60)

    results = {"LEFT (blue)": 0, "RIGHT (red)": 0}

    for run in range(args.runs):
        seed = args.seed if args.seed is not None else np.random.randint(0, 10000)

        print(f"\nRun {run + 1}/{args.runs} (seed={seed})")
        print("-" * 40)

        # Plan to TSR union
        import time
        t0 = time.perf_counter()
        path = planner.plan(start, [left_tsr, right_tsr], seed=seed)
        plan_time = time.perf_counter() - t0

        if path is None:
            print("No path found!")
            continue

        # Identify which goal was reached
        final_pose = robot.forward_kinematics(path[-1])
        goal_label = identify_reached_goal(final_pose, left_tsr, right_tsr)
        results[goal_label] += 1

        print(f"Found path: {len(path)} waypoints in {plan_time:.2f}s")
        print(f"Reached: {goal_label}")
        print(f"Final EE pos: {final_pose[:3, 3]}")

        # Visualize (only on last run if multiple)
        if run == args.runs - 1:
            if args.interactive:
                visualize_interactive(model, data, path, joints, goal_label)
            else:
                render_loop_video(
                    model, data, path, joints,
                    num_loops=args.loops,
                    output_path=args.output,
                    goal_label=goal_label,
                )

    # Summary for multiple runs
    if args.runs > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for label, count in results.items():
            pct = 100 * count / args.runs
            print(f"  {label}: {count}/{args.runs} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
