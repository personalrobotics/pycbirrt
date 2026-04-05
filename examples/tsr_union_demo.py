# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Demo: TSR Union planning with multiple grasp approaches.

This example demonstrates planning to a union of goal TSRs - the planner
finds a path to ANY goal in the set. We create two grasp approaches for
a single target cylinder:
1. TOP-DOWN: gripper pointing down from above
2. SIDE: gripper pointing horizontally from the front

The robot (UR5e + Robotiq 2F85) plans multiple grasps to a moving target,
showcasing both grasp approaches in an impressive looping video.

Prerequisites:
    export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie
    uv pip install pycbirrt[mujoco] mediapy

Usage:
    python examples/tsr_union_demo.py                    # Render to video
    python examples/tsr_union_demo.py --grasps 6         # 6 grasp cycles
    python examples/tsr_union_demo.py --output my.mp4    # Custom output
    mjpython examples/tsr_union_demo.py --interactive    # Interactive viewer
"""

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
from tsr import TSR

from pycbirrt import CBiRRT, CBiRRTConfig
from pycbirrt.backends.mujoco import (
    MuJoCoCollisionChecker,
    MuJoCoIKSolver,
    MuJoCoRobotModel,
)

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
        raise EnvironmentError("Set MUJOCO_MENAGERIE_PATH to your mujoco_menagerie clone")
    return Path(path)


def create_scene(menagerie_path: Path, target_pos: np.ndarray) -> "mujoco.MjModel":
    """Create scene with UR5e + Robotiq 2F85 gripper and a target cylinder."""
    ur5e_path = menagerie_path / "universal_robots_ur5e" / "ur5e.xml"
    robotiq_path = menagerie_path / "robotiq_2f85" / "2f85.xml"

    if not ur5e_path.exists():
        raise FileNotFoundError(f"UR5e model not found at {ur5e_path}")
    if not robotiq_path.exists():
        raise FileNotFoundError(f"Robotiq model not found at {robotiq_path}")

    # Load both models
    ur5e_spec = mujoco.MjSpec.from_file(str(ur5e_path))
    gripper_spec = mujoco.MjSpec.from_file(str(robotiq_path))

    # High-res offscreen rendering
    ur5e_spec.visual.global_.offwidth = 1920
    ur5e_spec.visual.global_.offheight = 1080

    # Attach gripper to UR5e
    wrist_body = ur5e_spec.body("wrist_3_link")
    attachment_site = None
    for site in wrist_body.sites:
        if site.name == "attachment_site":
            attachment_site = site
            break

    if attachment_site is None:
        raise RuntimeError("Could not find attachment_site on wrist_3_link")

    gripper_base = gripper_spec.body("base_mount")
    frame = wrist_body.add_frame()
    frame.name = "gripper_attachment_frame"
    frame.pos = attachment_site.pos
    frame.quat = attachment_site.quat
    frame.attach_body(gripper_base, "gripper_", "")

    world = ur5e_spec.worldbody

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

    # Table
    table = world.add_body()
    table.name = "table"
    table.pos = [0.5, 0, 0.30]

    table_top = table.add_geom()
    table_top.name = "table_top"
    table_top.type = mujoco.mjtGeom.mjGEOM_BOX
    table_top.size = [0.25, 0.35, 0.02]
    table_top.rgba = [0.45, 0.4, 0.35, 1]

    # Table legs
    for i, (x, y) in enumerate([(0.2, 0.30), (0.2, -0.30), (-0.2, 0.30), (-0.2, -0.30)]):
        leg = table.add_geom()
        leg.name = f"leg_{i}"
        leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        leg.pos = [x, y, -0.14]
        leg.size = [0.02, 0.12, 0]
        leg.rgba = [0.35, 0.3, 0.25, 1]
        leg.contype = 0
        leg.conaffinity = 0

    # Target cylinder (red)
    target = world.add_body()
    target.name = "target"
    target.pos = list(target_pos)

    target_geom = target.add_geom()
    target_geom.name = "target_geom"
    target_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    target_geom.size = [0.03, 0.06, 0]  # radius=3cm, half-height=6cm
    target_geom.rgba = [0.85, 0.2, 0.15, 1]  # Red

    return ur5e_spec.compile()


def create_grasp_tsrs(target_pos: np.ndarray) -> tuple[TSR, TSR]:
    """Create two grasp TSRs: top-down and side approach.

    The UR5e with Robotiq 2F85 gripper has:
    - attachment_site at the wrist flange
    - Gripper fingertips ~0.145m from base_mount
    - Total reach from attachment_site to fingertips ~0.145m

    Args:
        target_pos: Center position of the target cylinder

    Returns:
        (top_tsr, side_tsr) - two different grasp approaches
    """
    # Cylinder half-height is 0.06m, so top is at target_pos[2] + 0.06
    # For attachment_site positioning:
    # - Cylinder top at target_pos[2] + 0.06
    # - Clearance above cylinder: 0.02m
    # - Gripper length: 0.145m
    # - Total: standoff = 0.06 + 0.02 + 0.145 = 0.225m

    # === TOP-DOWN GRASP ===
    # Gripper approaches from above, z-axis pointing DOWN
    # Full yaw freedom (rotate around cylinder's vertical axis)
    standoff_top = 0.25  # Height above cylinder center (accounts for gripper length)
    T_top = np.eye(4)
    T_top[:3, 3] = target_pos + np.array([0, 0, standoff_top])
    # Rotation: gripper z points DOWN (180 deg rotation around x)
    T_top[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    top_tsr = TSR(
        T0_w=T_top,
        Tw_e=np.eye(4),
        Bw=np.array(
            [
                [-0.03, 0.03],  # x tolerance
                [-0.03, 0.03],  # y tolerance
                [0, 0.08],  # z: can be higher (more room for IK)
                [-0.02, 0.02],  # roll: small tolerance
                [-0.02, 0.02],  # pitch: small tolerance
                [-np.pi, np.pi],  # yaw: full rotation around vertical (approach from any direction)
            ]
        ),
    )

    # === SIDE GRASP ===
    # Gripper approaches horizontally from any direction around the cylinder.
    # The gripper z-axis points horizontally toward the cylinder center.
    # Full yaw freedom allows approaching from any angle around the cylinder.
    standoff_side = 0.20  # Distance from cylinder center

    # TSR frame at cylinder center
    T_side = np.eye(4)
    T_side[:3, 3] = target_pos

    # Tw_e: places gripper horizontally offset from TSR frame
    # At yaw=0, gripper is at +X offset from TSR origin, pointing toward -X (origin)
    # For a HORIZONTAL side grasp with fingers opening horizontally:
    #   gripper z-axis (approach) -> -X (toward cylinder)
    #   gripper y-axis (up) -> +Z (world up)
    #   gripper x-axis -> +Y (to the side)
    Tw_e_side = np.array(
        [
            [0, 0, -1, standoff_side],  # world X = -gripper_z, translation +X
            [1, 0, 0, 0],  # world Y = gripper_x
            [0, 1, 0, 0],  # world Z = gripper_y
            [0, 0, 0, 1],
        ]
    )

    side_tsr = TSR(
        T0_w=T_side,
        Tw_e=Tw_e_side,
        Bw=np.array(
            [
                [0, 0.06],  # x: can be further back
                [-0.03, 0.03],  # y tolerance
                [-0.03, 0.03],  # z tolerance
                [-0.1, 0.1],  # roll: some tolerance
                [-0.1, 0.1],  # pitch: some tolerance
                [-np.pi, np.pi],  # yaw: full 360 deg freedom around cylinder
            ]
        ),
    )

    return top_tsr, side_tsr


def identify_grasp_type(ee_pose: np.ndarray, top_tsr: TSR, side_tsr: TSR) -> str:
    """Identify which grasp approach was used."""
    top_dist, _ = top_tsr.distance(ee_pose)
    side_dist, _ = side_tsr.distance(ee_pose)

    if top_dist < side_dist:
        return "TOP-DOWN"
    else:
        return "SIDE"


def render_multi_grasp_video(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    grasp_cycles: list[dict],  # Each: {path, target_pos, grasp_type}
    joint_names: list[str],
    output_path: str,
    fps: int = 60,
    width: int = 1280,
    height: int = 720,
):
    """Render a video with multiple grasp cycles to different positions."""
    import mediapy as media

    renderer = mujoco.Renderer(model, width=width, height=height)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.35, 0.0, 0.35]
    camera.distance = 1.4
    camera.azimuth = 135  # Angle to see both top and side approaches
    camera.elevation = -25

    scene_option = mujoco.MjvOption()

    frames = []
    steps_per_waypoint = 3  # Smooth movement

    # Get target body ID for moving it
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")

    def set_target_pos(pos: np.ndarray):
        """Set target position by modifying model.body_pos (persists through mj_forward)."""
        model.body_pos[target_body_id] = pos

    def set_robot_config(q: np.ndarray):
        """Set robot joint configuration."""
        for j, jnt in enumerate(joint_names):
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt)
            data.qpos[model.jnt_qposadr[jnt_id]] = q[j]

    def render_frame():
        """Run forward kinematics and render a frame."""
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        frames.append(renderer.render())

    for cycle_idx, cycle in enumerate(grasp_cycles):
        path_forward = cycle["path"]
        target_pos = cycle["target_pos"]

        # Move target to new position (modifying model.body_pos persists through mj_forward)
        set_target_pos(target_pos)

        # Forward path (to grasp)
        for i in range(len(path_forward) - 1):
            for t in np.linspace(0, 1, steps_per_waypoint, endpoint=False):
                q = (1 - t) * path_forward[i] + t * path_forward[i + 1]
                set_robot_config(q)
                render_frame()

        # Hold at grasp position
        for _ in range(fps // 4):
            render_frame()

        # Backward path (return home)
        path_backward = list(reversed(path_forward))
        for i in range(len(path_backward) - 1):
            for t in np.linspace(0, 1, steps_per_waypoint, endpoint=False):
                q = (1 - t) * path_backward[i] + t * path_backward[i + 1]
                set_robot_config(q)
                render_frame()

        # Brief pause at home before next cycle
        for _ in range(fps // 6):
            render_frame()

    media.write_video(output_path, frames, fps=fps)

    # Summary
    top_count = sum(1 for c in grasp_cycles if c["grasp_type"] == "TOP-DOWN")
    side_count = sum(1 for c in grasp_cycles if c["grasp_type"] == "SIDE")

    print(f"\nSaved video to {output_path}")
    print(f"  {len(grasp_cycles)} grasp cycles, {len(frames)} frames, ~{len(frames) / fps:.1f}s")
    print(f"  Grasps: {top_count} TOP-DOWN, {side_count} SIDE")


def visualize_interactive(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    path_forward: list[np.ndarray],
    joint_names: list[str],
    goal_label: str,
):
    """Interactive viewer with looping playback."""
    import mujoco.viewer

    print(f"Opening viewer (planner chose {goal_label} grasp)")
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


def generate_target_positions(num_positions: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Generate random target positions on the table.

    Positions are chosen to be reachable for both TOP-DOWN and SIDE grasps.
    """
    positions = []
    # Table is centered at (0.5, 0, 0.30), top at z=0.32, size 0.25 x 0.35
    # Cylinder half-height is 0.06, so center is at z=0.38
    # For SIDE grasps to work, targets need clearance in front (negative x direction)
    for _ in range(num_positions):
        x = rng.uniform(0.40, 0.55)  # Moderate distance - reachable for both
        y = rng.uniform(-0.15, 0.15)  # Centered on table
        z = 0.38  # On table surface
        positions.append(np.array([x, y, z]))
    return positions


def main():
    parser = argparse.ArgumentParser(description="TSR Union demo: multiple grasp approaches")
    parser.add_argument("--output", "-o", default="/tmp/tsr_union_demo.mp4", help="Output video path")
    parser.add_argument("--grasps", type=int, default=6, help="Number of grasp cycles to show")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (None = random each run)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Use interactive viewer instead of video")
    args = parser.parse_args()

    menagerie = get_menagerie_path()

    # Initial target position (will be moved during planning)
    initial_target = np.array([0.45, 0.0, 0.38])

    print("Creating scene with UR5e + Robotiq 2F85 gripper...")
    model = create_scene(menagerie, initial_target)
    data = mujoco.MjData(model)
    print("Scene created successfully!")

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
            angular_joints=(True, True, True, True, True, True),
        )
    else:
        print("Using MuJoCo differential IK solver")
        ik_solver = MuJoCoIKSolver(model, data, "attachment_site", joints, collision)
        # Differential solver may need more pose samples
        config = CBiRRTConfig(
            timeout=30.0,
            goal_bias=0.15,
            tsr_samples=100,
            angular_joints=(True, True, True, True, True, True),
        )

    planner = CBiRRT(robot, ik_solver, collision, config)

    # Start configuration (home)
    start = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])

    print("\n" + "=" * 60)
    print("TSR UNION DEMO: Multiple Grasp Approaches")
    print("=" * 60)
    print("Robot: UR5e + Robotiq 2F85 gripper")
    print("Two grasp approaches:")
    print("  1. TOP-DOWN: gripper pointing down from above")
    print("  2. SIDE: gripper pointing horizontally from front")
    print(f"\nPlanning {args.grasps} grasps to moving targets...")
    print("=" * 60)

    # Generate random target positions
    master_seed = args.seed if args.seed is not None else np.random.randint(0, 10000)
    rng = np.random.default_rng(master_seed)
    target_positions = generate_target_positions(args.grasps, rng)

    # Plan all grasp cycles
    grasp_cycles = []
    import time

    for i, target_pos in enumerate(target_positions):
        seed = rng.integers(0, 10000)

        print(f"\nGrasp {i + 1}/{args.grasps}")
        print(f"  Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")

        # Create grasp TSRs for this target position
        top_tsr, side_tsr = create_grasp_tsrs(target_pos)

        # Plan to EITHER top-down or side grasp (TSR union)
        t0 = time.perf_counter()
        path = planner.plan(start, goal_tsrs=[top_tsr, side_tsr], seed=seed)
        plan_time = time.perf_counter() - t0

        if path is None:
            print("  No path found! Skipping...")
            continue

        final_pose = robot.forward_kinematics(path[-1])
        grasp_type = identify_grasp_type(final_pose, top_tsr, side_tsr)

        print(f"  Found: {len(path)} waypoints in {plan_time:.2f}s")
        print(f"  Grasp: {grasp_type}")

        grasp_cycles.append(
            {
                "path": path,
                "target_pos": target_pos,
                "grasp_type": grasp_type,
            }
        )

    if not grasp_cycles:
        print("\nNo paths found!")
        return

    # Summary
    top_count = sum(1 for c in grasp_cycles if c["grasp_type"] == "TOP-DOWN")
    side_count = sum(1 for c in grasp_cycles if c["grasp_type"] == "SIDE")
    print("\n" + "=" * 60)
    print("PLANNING COMPLETE")
    print(f"  {len(grasp_cycles)} successful grasps")
    print(f"  {top_count} TOP-DOWN, {side_count} SIDE")
    print("=" * 60)

    # Visualize
    if args.interactive:
        # For interactive mode, just show the first grasp cycle looping
        visualize_interactive(model, data, grasp_cycles[0]["path"], joints, grasp_cycles[0]["grasp_type"])
    else:
        render_multi_grasp_video(
            model,
            data,
            grasp_cycles,
            joints,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
