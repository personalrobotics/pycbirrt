# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Example: constrained transport with the UR5e in MuJoCo.

Carry something upright from one side of the robot to the other: the gripper
must point straight down along the whole path. This is CBiRRT's defining
case, a path constraint rather than a goal constraint, and the example shows
the difference by planning the same query twice, without and with the
constraint, and reporting the largest tilt of each path.

The start and goal are top-down grasp regions above two spots on the table,
front-right and back-left of the base, so the arm has to swing most of the
way around. The start region is deliberately awkward: each reachable pose
has several IK branches and only some are collision-free.

Usage:
    python examples/ur5e_transport.py --no-viz
    python examples/ur5e_transport.py --render transport.mp4
    python examples/ur5e_transport.py --ik mujoco --seed 3

Requires MuJoCo and MUJOCO_MENAGERIE_PATH (see ur5e_mujoco.py). SSIK is used
when installed; otherwise MuJoCo differential IK.
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from tsr import TSR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ur5e_mujoco import build_ik_solver, create_grasp_tsr, create_scene, get_menagerie_path  # noqa: E402

from pycbirrt import CBiRRT, CBiRRTConfig  # noqa: E402
from pycbirrt.backends.mujoco import MuJoCoCollisionChecker, MuJoCoRobotModel  # noqa: E402

UR5E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

START_POS = np.array([0.55, -0.35, 0.47])
GOAL_POS = np.array([-0.30, 0.45, 0.47])


def gripper_down_everywhere() -> TSR:
    """The path constraint: gripper z points down (roll and pitch within 0.05 rad), free yaw, above the table."""
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T[:3, 3] = [0.0, 0.0, 0.6]
    bounds = np.array([[-0.9, 0.9], [-0.9, 0.9], [-0.3, 0.5], [-0.05, 0.05], [-0.05, 0.05], [-np.pi, np.pi]])
    return TSR(T0_w=T, Tw_e=np.eye(4), Bw=bounds)


def max_tilt_violation(robot, path, upright: TSR) -> float:
    return max(upright.distance(robot.forward_kinematics(q))[0] for q in path)


def plan_transport(planner, robot, upright, seed, constrained: bool):
    kwargs = {"constraint_tsrs": [upright]} if constrained else {}
    t0 = time.perf_counter()
    result = planner.plan(
        start_tsrs=[create_grasp_tsr(START_POS)],
        goal_tsrs=[create_grasp_tsr(GOAL_POS)],
        seed=seed,
        return_details=True,
        **kwargs,
    )
    dt = time.perf_counter() - t0
    label = "constrained" if constrained else "unconstrained"
    if not result.success:
        print(f"  {label}: no path ({result.failure_reason})")
        return None
    tilt = max_tilt_violation(robot, result.path, upright)
    print(f"  {label}: {len(result.path)} waypoints in {dt:.2f}s, max tilt violation {tilt:.3f} rad")
    return result.path


def main():
    parser = argparse.ArgumentParser(description="UR5e constrained transport: keep the gripper pointing down")
    parser.add_argument("--render", type=str, help="Render the constrained path to a video file (e.g. transport.mp4)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--ik", choices=["auto", "ssik", "mujoco"], default="auto", help="IK backend (default: SSIK if installed)"
    )
    args = parser.parse_args()

    menagerie_path = get_menagerie_path()
    print("Creating scene with UR5e + Robotiq 2F85 gripper...")
    model = create_scene(menagerie_path)
    data = mujoco.MjData(model)

    robot = MuJoCoRobotModel(model, data, "attachment_site", UR5E_JOINTS)
    collision = MuJoCoCollisionChecker(model, data, UR5E_JOINTS)
    ik_solver, ik_name = build_ik_solver(model, data, UR5E_JOINTS, collision, menagerie_path, args.ik, seed=args.seed)
    print(f"Using {'SSIK (analytical)' if ik_name == 'ssik' else 'MuJoCo (differential)'} IK solver")

    # The UR5e's joints are bounded (±2π, elbow ±π), not continuous: leave angular_joints unset.
    config = CBiRRTConfig(max_iterations=5000, step_size=0.2, tsr_samples=100, timeout=60.0)
    planner = CBiRRT(robot, ik_solver, collision, config)
    upright = gripper_down_everywhere()

    print(f"\nTransport from {START_POS} to {GOAL_POS}, gripper down the whole way (seed {args.seed})")
    plan_transport(planner, robot, upright, args.seed, constrained=False)
    path = plan_transport(planner, robot, upright, args.seed, constrained=True)
    if path is None:
        return 1

    lo, hi = robot.joint_limits
    P = np.array(path)
    print(f"  every waypoint within joint limits: {bool(np.all((P >= lo) & (P <= hi)))}")
    print(f"  largest raw joint step: {np.abs(np.diff(P, axis=0)).max():.3f} rad (step size {config.step_size})")

    if args.render:
        from ur5e_mujoco import render_to_video

        render_to_video(model, data, path, UR5E_JOINTS, args.render)
    elif not args.no_viz:
        from ur5e_mujoco import visualize_interactive

        visualize_interactive(model, data, path, UR5E_JOINTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
