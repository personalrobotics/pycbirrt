"""Example: UR5e + Robotiq 2F85 planning with CBiRRT in MuJoCo.

This example demonstrates motion planning for a UR5e arm with a Robotiq 2F85
parallel jaw gripper to grasp a cylinder from a table. It uses TSR templates
to properly define:
1. The cylinder's placement on the table (object-to-surface relationship)
2. The gripper's grasp of the cylinder (gripper-to-object relationship)

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
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

from pycbirrt import CBiRRT, CBiRRTConfig
from pycbirrt.backends.mujoco import (
    MuJoCoRobotModel,
    MuJoCoCollisionChecker,
    MuJoCoIKSolver,
)
from tsr import TSR

# Optional EAIK import
try:
    from pycbirrt.backends.eaik import EAIKSolver
    EAIK_AVAILABLE = True
except ImportError:
    EAIK_AVAILABLE = False
from tsr.core.tsr_template import TSRTemplate
from tsr.schema import EntityClass, TaskCategory


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
# TSR Templates
# =============================================================================

def create_cylinder_on_table_template() -> TSRTemplate:
    """Create a TSR template for placing a cylinder on a table.

    This TSR defines valid poses for a cylinder sitting on a table surface.
    The reference frame is the table, and the subject is the cylinder.

    Returns:
        TSRTemplate for cylinder placement on table
    """
    # TSR frame is at the table surface, cylinder sits on top
    T_ref_tsr = np.eye(4)

    # Cylinder frame relative to TSR frame (cylinder base at table surface)
    # Cylinder is upright (z-axis up)
    Tw_e = np.eye(4)
    Tw_e[2, 3] = 0.05  # Cylinder half-height (0.05m) - center above surface

    # Bounds: cylinder can be anywhere on the table surface, any yaw rotation
    Bw = np.array([
        [-0.25, 0.25],    # x: table surface extent
        [-0.45, 0.45],    # y: table surface extent
        [0, 0],           # z: fixed on surface
        [0, 0],           # roll: upright
        [0, 0],           # pitch: upright
        [-np.pi, np.pi],  # yaw: any rotation around vertical
    ])

    return TSRTemplate(
        T_ref_tsr=T_ref_tsr,
        Tw_e=Tw_e,
        Bw=Bw,
        subject_entity=EntityClass.BOX,  # Using BOX as proxy for cylinder
        reference_entity=EntityClass.TABLE,
        task_category=TaskCategory.PLACE,
        variant="on",
        name="Cylinder on Table",
        description="Valid poses for a cylinder sitting upright on a table surface",
    )


def create_gripper_grasp_cylinder_template() -> TSRTemplate:
    """Create a TSR template for grasping a cylinder with a parallel jaw gripper.

    This TSR defines valid gripper poses for grasping a cylinder from above.
    The reference frame is the cylinder, and the subject is the gripper.

    The gripper approaches from above (top-down grasp), with the gripper's
    z-axis pointing down toward the cylinder. The grasp can be at any angle
    around the vertical axis (yaw freedom).

    For the UR5e, the attachment_site z-axis points along the tool direction
    (down when the arm is in home position), making top-down grasps natural.

    Returns:
        TSRTemplate for top-down grasping a cylinder
    """
    # TSR frame at cylinder center, aligned with world frame (z-up)
    T_ref_tsr = np.eye(4)

    # Gripper pose relative to TSR frame:
    # - Gripper z-axis points DOWN toward cylinder (approach direction)
    # - Gripper is positioned above the cylinder
    #
    # For UR5e with Robotiq 2F85 attached:
    # - attachment_site is at the wrist flange
    # - Gripper "pinch" site (fingertips) is ~0.145m from base_mount
    # - base_mount attaches at the attachment_site
    #
    # For a safe grasp approach:
    # - Cylinder half-height = 0.05m
    # - Gripper fingertips should stop ~5cm above cylinder top
    # - So attachment_site should be at:
    #   cylinder_center + 0.05 (to top) + 0.05 (clearance) + 0.145 (gripper length)
    #   = cylinder_center + 0.245m
    standoff = 0.25  # Height above cylinder center for attachment_site

    # Gripper frame: z points DOWN (for top-down approach)
    # We need a 180-degree rotation around x-axis to flip z from up to down
    # R_x(180) = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    #
    # With this rotation, the gripper's local z points DOWN in the TSR frame.
    # The standoff translation is along the gripper's z-axis (which points down),
    # so a positive value moves the gripper UP in world coordinates.
    Tw_e = np.array([
        [1, 0, 0, 0],            # gripper x unchanged
        [0, -1, 0, 0],           # gripper y flipped
        [0, 0, -1, standoff],    # gripper z flipped (points down), offset UP
        [0, 0, 0, 1],
    ])

    # Bounds: allow rotation around vertical axis, small xy tolerance
    # Note: rotation bounds are in the TSR frame where z points UP
    # Small roll/pitch tolerance accounts for numerical precision in IK
    Bw = np.array([
        [-0.02, 0.02],      # x: small tolerance
        [-0.02, 0.02],      # y: small tolerance
        [0, 0.05],          # z: can be 0-5cm higher (more room for collision avoidance)
        [-0.01, 0.01],      # roll: small tolerance for numerical precision
        [-0.01, 0.01],      # pitch: small tolerance for numerical precision
        [-np.pi, np.pi],    # yaw: any rotation around vertical
    ])

    return TSRTemplate(
        T_ref_tsr=T_ref_tsr,
        Tw_e=Tw_e,
        Bw=Bw,
        subject_entity=EntityClass.ROBOTIQ_2F140,  # Close enough to 2F85
        reference_entity=EntityClass.BOX,  # Using BOX as proxy for cylinder
        task_category=TaskCategory.GRASP,
        variant="top",
        name="Top-Down Grasp Cylinder",
        description="Grasp a cylinder from above with a parallel jaw gripper",
        preshape=np.array([0.07]),  # 7cm aperture to approach 6cm diameter cylinder
    )


# =============================================================================
# Scene Creation
# =============================================================================

def create_scene_model(menagerie_path: Path) -> "mujoco.MjModel":
    """Create a MuJoCo model with UR5e, Robotiq gripper, table, and cylinder.

    Uses MuJoCo's attach mechanism to connect the gripper to the robot arm.
    Returns the compiled model directly (no intermediate XML file needed).
    """
    import mujoco

    ur5e_path = menagerie_path / "universal_robots_ur5e" / "ur5e.xml"
    robotiq_path = menagerie_path / "robotiq_2f85" / "2f85.xml"

    if not ur5e_path.exists():
        raise FileNotFoundError(f"UR5e model not found at {ur5e_path}")
    if not robotiq_path.exists():
        raise FileNotFoundError(f"Robotiq model not found at {robotiq_path}")

    # Load both models
    ur5e_spec = mujoco.MjSpec.from_file(str(ur5e_path))
    gripper_spec = mujoco.MjSpec.from_file(str(robotiq_path))

    # Set offscreen framebuffer size for high-res rendering
    ur5e_spec.visual.global_.offwidth = 1920
    ur5e_spec.visual.global_.offheight = 1080

    # Find the wrist_3_link body on the UR5e where we'll attach the gripper
    wrist_body = ur5e_spec.body("wrist_3_link")

    # Find the attachment site
    attachment_site = None
    for site in wrist_body.sites:
        if site.name == "attachment_site":
            attachment_site = site
            break

    if attachment_site is None:
        raise RuntimeError("Could not find attachment_site on wrist_3_link")

    # Find the gripper's base body
    gripper_base = gripper_spec.body("base_mount")

    # Attach the gripper to the UR5e at the attachment site
    frame = wrist_body.add_frame()
    frame.name = "gripper_attachment_frame"
    frame.pos = attachment_site.pos
    frame.quat = attachment_site.quat
    # Use "gripper_" prefix to avoid name collisions (both models have "black" material)
    frame.attach_body(gripper_base, "gripper_", "")

    # Add scene elements (ground, table, cylinder)
    world = ur5e_spec.worldbody

    # Ground plane - light gray for clean look
    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.pos = [0, 0, -0.01]
    floor.size = [0, 0, 0.05]
    floor.rgba = [0.95, 0.95, 0.95, 1]  # Light gray / near-white

    # Robot base marker (visual only)
    base_marker = world.add_geom()
    base_marker.name = "base_marker"
    base_marker.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    base_marker.pos = [0, 0, 0.01]
    base_marker.size = [0.1, 0.01, 0]
    base_marker.rgba = [0.3, 0.5, 0.7, 0.8]  # Subtle blue
    base_marker.contype = 0
    base_marker.conaffinity = 0

    # Table body
    table_body = world.add_body()
    table_body.name = "table"
    table_body.pos = [0.5, 0, 0.4]  # Table top at z=0.42

    # Table top (collision enabled)
    table_top = table_body.add_geom()
    table_top.name = "table_top"
    table_top.type = mujoco.mjtGeom.mjGEOM_BOX
    table_top.size = [0.25, 0.4, 0.02]
    table_top.rgba = [0.4, 0.3, 0.2, 1]

    # Table legs (visual, no collision)
    leg_positions = [
        [0.22, 0.37, -0.20],
        [0.22, -0.37, -0.20],
        [-0.22, 0.37, -0.20],
        [-0.22, -0.37, -0.20],
    ]
    for i, lpos in enumerate(leg_positions):
        leg = table_body.add_geom()
        leg.name = f"table_leg_{i}"
        leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        leg.pos = lpos
        leg.size = [0.025, 0.18, 0]
        leg.rgba = [0.3, 0.25, 0.2, 1]
        leg.contype = 0
        leg.conaffinity = 0

    # Target cylinder on table (collision enabled)
    cylinder_body = world.add_body()
    cylinder_body.name = "cylinder"
    cylinder_body.pos = [0.45, 0.15, 0.47]  # On table surface + half height

    cylinder_geom = cylinder_body.add_geom()
    cylinder_geom.name = "cylinder_geom"
    cylinder_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    cylinder_geom.size = [0.03, 0.05, 0]  # radius=3cm, half-height=5cm
    cylinder_geom.rgba = [1, 0.2, 0.2, 1]

    # Compile and return the model
    return ur5e_spec.compile()


def create_simple_scene_file(menagerie_path: Path) -> Path:
    """Create a simpler scene file with just UR5e (no gripper attachment).

    This is a fallback if the gripper attachment fails.
    """
    ur5e_dir = menagerie_path / "universal_robots_ur5e"
    ur5e_path = ur5e_dir / "ur5e.xml"

    if not ur5e_path.exists():
        raise FileNotFoundError(f"UR5e model not found at {ur5e_path}")

    # Collision is enabled on table and cylinder (no contype/conaffinity overrides)
    scene_xml = """
<mujoco model="ur5e_scene">
  <include file="ur5e.xml"/>

  <worldbody>
    <!-- Ground plane - light gray for clean look -->
    <geom name="floor" pos="0 0 -0.01" size="0 0 0.05" type="plane" rgba="0.95 0.95 0.95 1"/>

    <!-- Robot base marker (visual only) -->
    <geom name="base_marker" type="cylinder" pos="0 0 0.01" size="0.1 0.01"
          rgba="0.3 0.5 0.7 0.8" contype="0" conaffinity="0"/>

    <!-- Table in front of robot -->
    <body name="table" pos="0.5 0 0.4">
      <geom type="box" size="0.25 0.4 0.02" rgba="0.4 0.3 0.2 1" name="table_top"/>
      <!-- Table legs (visual only) -->
      <geom type="cylinder" size="0.025 0.18" pos="0.22 0.37 -0.20" rgba="0.3 0.25 0.2 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.025 0.18" pos="0.22 -0.37 -0.20" rgba="0.3 0.25 0.2 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.025 0.18" pos="-0.22 0.37 -0.20" rgba="0.3 0.25 0.2 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.025 0.18" pos="-0.22 -0.37 -0.20" rgba="0.3 0.25 0.2 1" contype="0" conaffinity="0"/>
    </body>

    <!-- Target cylinder on table (collision enabled) -->
    <body name="cylinder" pos="0.45 0.15 0.47">
      <geom type="cylinder" size="0.03 0.05" rgba="1 0.2 0.2 1" name="cylinder_geom"/>
    </body>
  </worldbody>
</mujoco>
"""
    scene_path = ur5e_dir / "pycbirrt_scene.xml"
    scene_path.write_text(scene_xml)
    return scene_path


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

    # Camera setup for good viewing angle
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.25, 0.0, 0.35]
    camera.distance = 1.4
    camera.azimuth = 135
    camera.elevation = -25

    # Scene options for white background
    scene_option = mujoco.MjvOption()

    frames = []
    steps_per_waypoint = 3  # Faster movement (was 10)

    for i in range(len(path) - 1):
        for t in np.linspace(0, 1, steps_per_waypoint, endpoint=False):
            q = (1 - t) * path[i] + t * path[i + 1]

            # Set joint positions
            for j, jnt_name in enumerate(joint_names):
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                qpos_adr = model.jnt_qposadr[jnt_id]
                data.qpos[qpos_adr] = q[j]

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
            print("Path planning succeeded!")
            print("Options:")
            print("  - On macOS: run with mjpython")
            print("  - Any platform: use --render video.mp4 to save a video")
            return
        raise

    with viewer:
        waypoint_idx = 0
        steps_per_waypoint = 15  # Faster movement (was 50)

        while viewer.is_running():
            if waypoint_idx < len(path) - 1:
                t = (data.time * 30) % steps_per_waypoint / steps_per_waypoint  # Faster (was *10)
                q = (1 - t) * path[waypoint_idx] + t * path[waypoint_idx + 1]

                if t > 0.99:
                    waypoint_idx += 1
            else:
                q = path[-1]

            for i, jnt_name in enumerate(joint_names):
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                qpos_adr = model.jnt_qposadr[jnt_id]
                data.qpos[qpos_adr] = q[i]

            mujoco.mj_forward(model, data)
            viewer.sync()
            data.time += 0.01


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="UR5e + Robotiq 2F85 motion planning with TSR-based goals"
    )
    parser.add_argument("--render", type=str, help="Render to video file (e.g., output.mp4)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--ik",
        type=str,
        choices=["mujoco", "eaik"],
        default="mujoco",
        help="IK solver to use: 'mujoco' (differential) or 'eaik' (analytical, requires eaik package)",
    )
    parser.add_argument(
        "--compare-ik",
        action="store_true",
        help="Compare both IK solvers on the same planning problem",
    )
    args = parser.parse_args()

    # Check EAIK availability
    if args.ik == "eaik" and not EAIK_AVAILABLE:
        print("Error: EAIK not available. Install with: pip install eaik")
        return
    if args.compare_ik and not EAIK_AVAILABLE:
        print("Error: EAIK not available for comparison. Install with: pip install eaik")
        return

    menagerie_path = get_menagerie_path()

    # Create MuJoCo model with UR5e + Robotiq 2F85 gripper
    print("Creating scene with UR5e + Robotiq 2F85 gripper...")
    try:
        model = create_scene_model(menagerie_path)
        print("Scene created successfully with gripper attached")
    except Exception as e:
        print(f"Warning: Failed to create gripper scene: {e}")
        print("Falling back to simple scene without gripper...")
        scene_path = create_simple_scene_file(menagerie_path)
        model = mujoco.MjModel.from_xml_path(str(scene_path))

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
    robot = MuJoCoRobotModel(
        model=model,
        data=data,
        ee_site="attachment_site",  # UR5e's end-effector site
        joint_names=ur5e_joints,
    )
    collision_checker = MuJoCoCollisionChecker(
        model=model,
        data=data,
        joint_names=ur5e_joints,
    )

    # Create IK solver based on selection
    if args.ik == "eaik":
        print("Using EAIK (analytical) IK solver")
        ik_solver = EAIKSolver.for_ur5e(
            joint_limits=robot.joint_limits,
            collision_checker=collision_checker,
        )
        print(f"  Kinematic family: {ik_solver.get_kinematic_family()}")
        # EAIK doesn't need multiple seeds - it finds all solutions analytically
        config = CBiRRTConfig(
            max_iterations=5000,
            step_size=0.2,
            goal_bias=0.1,
            ik_num_seeds=1,  # EAIK finds all solutions, one seed is enough
        )
    else:
        print("Using MuJoCo (differential) IK solver")
        ik_solver = MuJoCoIKSolver(
            model=model,
            data=data,
            ee_site="attachment_site",
            joint_names=ur5e_joints,
            collision_checker=collision_checker,
        )
        config = CBiRRTConfig(
            max_iterations=5000,
            step_size=0.2,
            goal_bias=0.1,
            ik_num_seeds=20,  # Try more IK seeds for differential solver
        )

    # Create planner
    planner = CBiRRT(
        robot=robot,
        ik_solver=ik_solver,
        collision_checker=collision_checker,
        config=config,
    )

    # =========================================================================
    # Define the manipulation task using TSR templates
    # =========================================================================

    # 1. Create TSR templates for the task
    placement_template = create_cylinder_on_table_template()
    grasp_template = create_gripper_grasp_cylinder_template()

    print("TSR Templates:")
    print(f"  - {placement_template.name}: {placement_template.description}")
    print(f"  - {grasp_template.name}: {grasp_template.description}")

    # 2. Get world poses for the entities
    # Table pose (table top surface is at z=0.42 since table body is at 0.4, box half-height is 0.02)
    T_table_world = np.eye(4)
    T_table_world[:3, 3] = [0.5, 0.0, 0.42]  # Table surface position

    # Cylinder pose (center of cylinder)
    # Cylinder is at (0.45, 0.15, 0.47) with half-height 0.05
    # This matches the scene creation in create_scene_file()
    T_cylinder_world = np.eye(4)
    T_cylinder_world[:3, 3] = [0.45, 0.15, 0.47]  # Cylinder center position

    # 3. Instantiate the grasp TSR at the cylinder's current pose
    # This gives us valid gripper poses for grasping this specific cylinder
    grasp_tsr = grasp_template.instantiate(T_cylinder_world)

    print(f"\nCylinder at: {T_cylinder_world[:3, 3]}")
    print(f"Grasp TSR frame at: {grasp_tsr.T0_w[:3, 3]}")

    # 4. Sample a few grasp poses to verify the TSR is correct
    print("\nSample grasp poses from TSR:")
    for i in range(3):
        sample_pose = grasp_tsr.sample()
        print(f"  Sample {i+1}: position={sample_pose[:3, 3]}")

    # =========================================================================
    # Plan motion to grasp pose
    # =========================================================================

    # Start configuration (home position)
    start = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])

    # =========================================================================
    # IK Comparison Mode
    # =========================================================================
    if args.compare_ik:
        import time

        print("\n" + "=" * 70)
        print("IK SOLVER COMPARISON")
        print("=" * 70)

        # Sample some target poses from the grasp TSR
        n_samples = 10
        print(f"\nSampling {n_samples} target poses from grasp TSR...")
        target_poses = [grasp_tsr.sample() for _ in range(n_samples)]

        # Create both solvers
        mujoco_solver = MuJoCoIKSolver(
            model=model,
            data=data,
            ee_site="attachment_site",
            joint_names=ur5e_joints,
            collision_checker=collision_checker,
        )
        eaik_solver = EAIKSolver.for_ur5e(
            joint_limits=robot.joint_limits,
            collision_checker=collision_checker,
        )

        print(f"\nEAIK Kinematic family: {eaik_solver.get_kinematic_family()}")

        # Compare IK solving
        print(f"\n{'Solver':<15} {'Success':<10} {'Avg Time (ms)':<15} {'Solutions/pose':<15}")
        print("-" * 55)

        # EAIK benchmark
        eaik_successes = 0
        eaik_total_solutions = 0
        eaik_start = time.perf_counter()
        for pose in target_poses:
            solutions = eaik_solver.solve_valid(pose)
            if solutions:
                eaik_successes += 1
                eaik_total_solutions += len(solutions)
        eaik_time = (time.perf_counter() - eaik_start) * 1000

        # MuJoCo benchmark (with multiple seeds)
        mujoco_successes = 0
        mujoco_total_solutions = 0
        mujoco_start = time.perf_counter()
        lower, upper = robot.joint_limits
        rng = np.random.default_rng(args.seed)
        for pose in target_poses:
            found = False
            for _ in range(20):  # Try 20 random seeds
                q_init = rng.uniform(lower, upper)
                solutions = mujoco_solver.solve_valid(pose, q_init)
                if solutions:
                    mujoco_successes += 1
                    mujoco_total_solutions += 1  # Differential IK returns at most 1
                    found = True
                    break
        mujoco_time = (time.perf_counter() - mujoco_start) * 1000

        print(f"{'EAIK':<15} {eaik_successes}/{n_samples:<9} {eaik_time:.2f}{'':>10} {eaik_total_solutions/max(1,eaik_successes):.1f}")
        print(f"{'MuJoCo':<15} {mujoco_successes}/{n_samples:<9} {mujoco_time:.2f}{'':>10} {mujoco_total_solutions/max(1,mujoco_successes):.1f}")

        print("\nNotes:")
        print("  - EAIK returns all analytical solutions (typically 8 for UR5e)")
        print("  - MuJoCo differential IK returns at most 1 solution per seed")
        print("  - EAIK time includes checking all solutions for validity")
        print("  - MuJoCo time includes trying up to 20 random initial configurations")
        print("=" * 70 + "\n")

    # =========================================================================
    # Plan motion to grasp pose
    # =========================================================================

    print("\nPlanning path from home to grasp pose...")
    import time
    plan_start = time.perf_counter()
    path = planner.plan(start, goal_tsrs=[grasp_tsr], seed=args.seed)
    plan_time = time.perf_counter() - plan_start

    if path is None:
        print("No path found!")
        return

    print(f"Found path with {len(path)} waypoints in {plan_time:.2f}s")

    # Verify goal reached
    final_pose = robot.forward_kinematics(path[-1])
    dist, bwopt = grasp_tsr.distance(final_pose)
    print(f"Final EE position: {final_pose[:3, 3]}")
    print(f"Final EE z-axis: {final_pose[:3, 2]}")
    print(f"Distance to TSR: {dist:.4f}")
    if dist > 0.1:
        print(f"  bwopt: {bwopt}")
        print(f"  TSR Bw: {grasp_tsr.Bw}")

    # Visualization
    if args.no_viz:
        print("Skipping visualization (--no-viz)")
    elif args.render:
        render_to_video(model, data, path, ur5e_joints, args.render)
    else:
        visualize_interactive(model, data, path, ur5e_joints)


if __name__ == "__main__":
    main()
