# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Example: UR5e + Robotiq 2F85 planning with CBiRRT in MuJoCo.

This example demonstrates motion planning for a UR5e arm with a Robotiq 2F85
parallel jaw gripper to grasp a cylinder from a table. A TSR defines the valid
top-down grasp region around the cylinder.

Prerequisites:
1. Clone MuJoCo Menagerie:
   git clone https://github.com/google-deepmind/mujoco_menagerie.git

2. Set environment variable:
   export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie

3. Install dependencies:
   uv pip install pycbirrt[mujoco] mediapy

Running:
- Default (interactive viewer):
    mjpython examples/ur5e_mujoco.py
- Render to video (no viewer required):
    python examples/ur5e_mujoco.py --render video.mp4
- Plan only (no visualization):
    python examples/ur5e_mujoco.py --no-viz
"""

import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from tsr import TSR

from pycbirrt import CBiRRT, CBiRRTConfig
from pycbirrt.backends.mujoco import (
    MuJoCoCollisionChecker,
    MuJoCoIKSolver,
    MuJoCoRobotModel,
)

# Optional EAIK import (analytical IK, faster than differential)
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
            "MUJOCO_MENAGERIE_PATH environment variable not set. "
            "Please clone https://github.com/google-deepmind/mujoco_menagerie "
            "and set MUJOCO_MENAGERIE_PATH to the clone location."
        )
    return Path(path)


# =============================================================================
# Grasp TSR
# =============================================================================


def create_grasp_tsr(target_pos: np.ndarray) -> TSR:
    """Create a TSR for top-down grasp of a cylinder.

    The gripper approaches from above with z-axis pointing DOWN.
    Full yaw freedom allows grasping from any angle around the vertical axis.

    Geometry: attachment_site is 0.25m above the cylinder center,
    accounting for gripper length (~0.145m) and clearance.
    """
    standoff = 0.25  # Height above cylinder center for attachment_site

    T0_w = np.eye(4)
    T0_w[:3, 3] = target_pos + np.array([0, 0, standoff])
    # Gripper z points DOWN (180 deg rotation around x)
    T0_w[:3, :3] = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )

    return TSR(
        T0_w=T0_w,
        Tw_e=np.eye(4),
        Bw=np.array(
            [
                [-0.02, 0.02],  # x tolerance
                [-0.02, 0.02],  # y tolerance
                [0, 0.05],  # z: can be 0-5cm higher
                [-0.01, 0.01],  # roll: small tolerance
                [-0.01, 0.01],  # pitch: small tolerance
                [-np.pi, np.pi],  # yaw: full rotation
            ]
        ),
    )


# =============================================================================
# Scene Creation
# =============================================================================


def create_scene(menagerie_path: Path) -> "mujoco.MjModel":
    """Create a MuJoCo model with UR5e, Robotiq gripper, table, and cylinder.

    Uses MuJoCo's attach mechanism to connect the gripper to the robot arm.
    """
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

    # Attach gripper to UR5e wrist
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
    # "gripper_" prefix avoids name collisions (both models have "black" material)
    frame.attach_body(gripper_base, "gripper_", "")

    # Scene elements
    world = ur5e_spec.worldbody

    # Ground plane
    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.pos = [0, 0, -0.01]
    floor.size = [0, 0, 0.05]
    floor.rgba = [0.95, 0.95, 0.95, 1]

    # Robot base marker (visual only)
    base_marker = world.add_geom()
    base_marker.name = "base_marker"
    base_marker.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    base_marker.pos = [0, 0, 0.01]
    base_marker.size = [0.1, 0.01, 0]
    base_marker.rgba = [0.3, 0.5, 0.7, 0.8]
    base_marker.contype = 0
    base_marker.conaffinity = 0

    # Table
    table_body = world.add_body()
    table_body.name = "table"
    table_body.pos = [0.5, 0, 0.4]  # Table top at z=0.42

    table_top = table_body.add_geom()
    table_top.name = "table_top"
    table_top.type = mujoco.mjtGeom.mjGEOM_BOX
    table_top.size = [0.25, 0.4, 0.02]
    table_top.rgba = [0.4, 0.3, 0.2, 1]

    # Table legs (visual only)
    for i, (x, y) in enumerate([(0.22, 0.37), (0.22, -0.37), (-0.22, 0.37), (-0.22, -0.37)]):
        leg = table_body.add_geom()
        leg.name = f"table_leg_{i}"
        leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        leg.pos = [x, y, -0.20]
        leg.size = [0.025, 0.18, 0]
        leg.rgba = [0.3, 0.25, 0.2, 1]
        leg.contype = 0
        leg.conaffinity = 0

    # Target cylinder on table
    cylinder_body = world.add_body()
    cylinder_body.name = "cylinder"
    cylinder_body.pos = [0.45, 0.15, 0.47]  # On table surface + half height

    cylinder_geom = cylinder_body.add_geom()
    cylinder_geom.name = "cylinder_geom"
    cylinder_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    cylinder_geom.size = [0.03, 0.05, 0]  # radius=3cm, half-height=5cm
    cylinder_geom.rgba = [1, 0.2, 0.2, 1]

    return ur5e_spec.compile()


# =============================================================================
# Visualization
# =============================================================================


def render_to_video(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    path: list[np.ndarray],
    joint_names: list[str],
    output_path: str,
    fps: int = 60,
    width: int = 1280,
    height: int = 720,
):
    """Render the planned path to a video file."""
    import mediapy as media

    renderer = mujoco.Renderer(model, width=width, height=height)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.25, 0.0, 0.35]
    camera.distance = 1.4
    camera.azimuth = 135
    camera.elevation = -25

    scene_option = mujoco.MjvOption()

    frames = []
    steps_per_waypoint = 3

    for i in range(len(path) - 1):
        for t in np.linspace(0, 1, steps_per_waypoint, endpoint=False):
            q = (1 - t) * path[i] + t * path[i + 1]

            for j, jnt_name in enumerate(joint_names):
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                data.qpos[model.jnt_qposadr[jnt_id]] = q[j]

            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            frames.append(renderer.render())

    # Hold final frame briefly
    for _ in range(fps // 2):
        frames.append(renderer.render())

    media.write_video(output_path, frames, fps=fps)
    print(f"Saved video to {output_path}")


def visualize_interactive(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    path: list[np.ndarray],
    joint_names: list[str],
):
    """Visualize the path in an interactive viewer."""
    print("Opening viewer. Press ESC to close.")
    print("(On macOS, run with: mjpython examples/ur5e_mujoco.py)")

    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        if "mjpython" in str(e):
            print(f"Viewer not available: {e}")
            print("Use --render video.mp4 to save a video instead")
            return
        raise

    with viewer:
        waypoint_idx = 0
        steps_per_waypoint = 15

        while viewer.is_running():
            if waypoint_idx < len(path) - 1:
                t = (data.time * 30) % steps_per_waypoint / steps_per_waypoint
                q = (1 - t) * path[waypoint_idx] + t * path[waypoint_idx + 1]

                if t > 0.99:
                    waypoint_idx += 1
            else:
                q = path[-1]

            for i, jnt_name in enumerate(joint_names):
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                data.qpos[model.jnt_qposadr[jnt_id]] = q[i]

            mujoco.mj_forward(model, data)
            viewer.sync()
            data.time += 0.01


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="UR5e + Robotiq 2F85 motion planning with TSR-based goals")
    parser.add_argument("--render", type=str, help="Render to video file (e.g., output.mp4)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    menagerie_path = get_menagerie_path()

    # Create scene
    print("Creating scene with UR5e + Robotiq 2F85 gripper...")
    model = create_scene(menagerie_path)
    data = mujoco.MjData(model)

    # UR5e joint names
    ur5e_joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # Create robot model and collision checker
    robot = MuJoCoRobotModel(model, data, "attachment_site", ur5e_joints)
    collision_checker = MuJoCoCollisionChecker(model, data, ur5e_joints)

    # IK solver: prefer EAIK (analytical, faster), fall back to MuJoCo (differential)
    if EAIK_AVAILABLE:
        print("Using EAIK (analytical) IK solver")
        ik_solver = EAIKSolver.for_ur5e(robot.joint_limits, collision_checker)
    else:
        print("Using MuJoCo (differential) IK solver (install eaik for faster planning)")
        ik_solver = MuJoCoIKSolver(model, data, "attachment_site", ur5e_joints, collision_checker)

    config = CBiRRTConfig(
        max_iterations=5000,
        step_size=0.2,
        goal_bias=0.1,
        tsr_samples=100,
        angular_joints=(True, True, True, True, True, True),
    )

    planner = CBiRRT(robot, ik_solver, collision_checker, config)

    # Target cylinder position (matches scene creation)
    target_pos = np.array([0.45, 0.15, 0.47])
    grasp_tsr = create_grasp_tsr(target_pos)

    print(f"\nCylinder at: {target_pos}")
    print(f"Grasp TSR frame at: {grasp_tsr.T0_w[:3, 3]}")

    # Start configuration (home position)
    start = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])

    # Plan
    print("\nPlanning path from home to grasp pose...")
    t0 = time.perf_counter()
    path = planner.plan(start, goal_tsrs=[grasp_tsr], seed=args.seed)
    plan_time = time.perf_counter() - t0

    if path is None:
        print("No path found!")
        return

    print(f"Found path with {len(path)} waypoints in {plan_time:.2f}s")

    # Verify goal reached
    final_pose = robot.forward_kinematics(path[-1])
    dist, _ = grasp_tsr.distance(final_pose)
    print(f"Final EE position: {final_pose[:3, 3]}")
    print(f"Distance to TSR: {dist:.4f}")

    # Visualization
    if args.no_viz:
        print("Skipping visualization (--no-viz)")
    elif args.render:
        render_to_video(model, data, path, ur5e_joints, args.render)
    else:
        visualize_interactive(model, data, path, ur5e_joints)


if __name__ == "__main__":
    main()
