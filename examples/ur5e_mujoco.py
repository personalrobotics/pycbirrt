"""Example: UR5e planning with CBiRRT in MuJoCo.

Prerequisites:
1. Clone MuJoCo Menagerie:
   git clone https://github.com/google-deepmind/mujoco_menagerie.git

2. Set environment variable:
   export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie

3. Install dependencies:
   uv pip install pycbirrt[mujoco]
"""

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


def create_scene_xml(menagerie_path: Path) -> str:
    """Create a scene XML that combines UR5e with Robotiq gripper and a table."""
    ur5e_path = menagerie_path / "universal_robots_ur5e" / "ur5e.xml"
    robotiq_path = menagerie_path / "robotiq_2f85" / "2f85.xml"

    # Verify files exist
    if not ur5e_path.exists():
        raise FileNotFoundError(f"UR5e model not found at {ur5e_path}")
    if not robotiq_path.exists():
        raise FileNotFoundError(f"Robotiq model not found at {robotiq_path}")

    return f"""
<mujoco model="ur5e_robotiq_scene">
  <include file="{ur5e_path}"/>

  <worldbody>
    <!-- Table -->
    <body name="table" pos="0.5 0 0.4">
      <geom type="box" size="0.4 0.6 0.02" rgba="0.4 0.3 0.2 1" name="table_top"/>
    </body>

    <!-- Target object -->
    <body name="target" pos="0.5 0.2 0.45">
      <geom type="cylinder" size="0.03 0.05" rgba="1 0 0 1" name="target_geom"/>
    </body>
  </worldbody>

  <!-- Attach gripper to UR5e end-effector -->
  <attach>
    <body1 name="wrist_3_link"/>
    <prefix name="gripper_"/>
    <include file="{robotiq_path}"/>
  </attach>

  <!-- End-effector site for IK -->
  <worldbody>
    <site name="ee_site" pos="0 0 0.15" parent="wrist_3_link"/>
  </worldbody>
</mujoco>
"""


def create_scene_file(menagerie_path: Path) -> Path:
    """Create a scene file with UR5e and obstacles.

    We need to write the scene file in the UR5e directory because MuJoCo
    resolves meshdir paths relative to the including file, not the included file.
    """
    ur5e_dir = menagerie_path / "universal_robots_ur5e"
    ur5e_path = ur5e_dir / "ur5e.xml"

    if not ur5e_path.exists():
        raise FileNotFoundError(f"UR5e model not found at {ur5e_path}")

    scene_xml = """
<mujoco model="ur5e_scene">
  <include file="ur5e.xml"/>

  <worldbody>
    <!-- Table -->
    <body name="table" pos="0.5 0 0.3">
      <geom type="box" size="0.4 0.6 0.02" rgba="0.4 0.3 0.2 1" name="table_top"/>
    </body>

    <!-- Target object -->
    <body name="target" pos="0.4 0.2 0.35">
      <geom type="cylinder" size="0.03 0.05" rgba="1 0 0 1" name="target_geom"/>
    </body>
  </worldbody>
</mujoco>
"""
    # Write scene file in the UR5e directory so relative mesh paths work
    scene_path = ur5e_dir / "pycbirrt_scene.xml"
    scene_path.write_text(scene_xml)
    return scene_path


def main():
    menagerie_path = get_menagerie_path()

    # Create scene file
    scene_path = create_scene_file(menagerie_path)

    # Load MuJoCo model from file (needed for mesh path resolution)
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
        ee_site="attachment_site",  # UR5e's end-effector site from menagerie
        joint_names=ur5e_joints,
    )
    collision_checker = MuJoCoCollisionChecker(
        model=model,
        data=data,
        joint_names=ur5e_joints,
    )

    # Create IK solver using MuJoCo's differential IK
    ik_solver = MuJoCoIKSolver(
        model=model,
        data=data,
        ee_site="attachment_site",
        joint_names=ur5e_joints,
        collision_checker=collision_checker,
    )

    # Create planner
    config = CBiRRTConfig(
        max_iterations=5000,
        step_size=0.2,
        goal_bias=0.1,
    )
    planner = CBiRRT(
        robot=robot,
        ik_solver=ik_solver,
        collision_checker=collision_checker,
        config=config,
    )

    # Define goal TSR (reach above the target object)
    target_pos = np.array([0.4, 0.2, 0.45])  # Above the cylinder
    T0_w = np.eye(4)
    T0_w[:3, 3] = target_pos

    # End-effector pointing down
    Tw_e = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ])

    goal_tsr = TSR(
        T0_w=T0_w,
        Tw_e=Tw_e,
        Bw=np.array([
            [-0.02, 0.02],   # x tolerance
            [-0.02, 0.02],   # y tolerance
            [0.0, 0.05],     # z: 0-5cm above
            [0, 0],          # roll: fixed
            [0, 0],          # pitch: fixed
            [-np.pi, np.pi], # yaw: free rotation
        ]),
    )

    # Start configuration (home position)
    start = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])

    print("Planning path from home to target...")
    path = planner.plan(start, goal_tsrs=[goal_tsr], seed=42)

    if path is None:
        print("No path found!")
        return

    print(f"Found path with {len(path)} waypoints")

    # Verify goal reached
    final_pose = robot.forward_kinematics(path[-1])
    print(f"Final EE position: {final_pose[:3, 3]}")
    print(f"Target position: {target_pos}")

    # Visualize in MuJoCo viewer
    # Note: On macOS, must run with `mjpython` instead of `python`
    print("Opening viewer. Press ESC to close.")
    print("(On macOS, run with: mjpython examples/ur5e_mujoco.py)")
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        if "mjpython" in str(e):
            print(f"Viewer not available: {e}")
            print("Path planning succeeded! Run with mjpython to see visualization.")
            return
        raise

    with viewer:
        # Animate the path
        waypoint_idx = 0
        steps_per_waypoint = 50

        while viewer.is_running():
            # Interpolate between waypoints
            if waypoint_idx < len(path) - 1:
                t = (data.time * 10) % steps_per_waypoint / steps_per_waypoint
                q = (1 - t) * path[waypoint_idx] + t * path[waypoint_idx + 1]

                if t > 0.99:
                    waypoint_idx += 1
            else:
                q = path[-1]

            # Set joint positions
            for i, jnt_name in enumerate(ur5e_joints):
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                qpos_adr = model.jnt_qposadr[jnt_id]
                data.qpos[qpos_adr] = q[i]

            mujoco.mj_forward(model, data)
            viewer.sync()

            # Step time
            data.time += 0.01


if __name__ == "__main__":
    main()
