# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Example: Franka CBiRRT planning to a live-editable TSR goal, in viser.

A Task Space Region (``tsr.TSR``) describes a *goal region* for the end-effector
rather than a single goal pose: a frame ``T0_w`` plus a 6-DoF box ``Bw`` of
allowed offsets in the CGA-split coordinates ``[tx, ty, tz, b12, b13, b23]``
(translation + rotor-bivector log). The CBiRRT planner samples poses from the
region and drives the arm into it.

This demo plans the Franka into a top-down grasp TSR in front of the robot and
exposes the region's ``Bw`` extents as viser sliders. Whenever you change a
slider the TSR is rebuilt and the planner **replans** to the new region (the
serve loop debounces the slider stream so it only replans once the drag settles).

Everything is gafro-CGA-native: a ``gafro.System`` does FK/Jacobians via its
arm task space, the same ``System`` carries the meshes the ``Visualizer`` drives
with FK, and the TSR geometry is gafro Motors throughout -- no MuJoCo / EAIK.

Install the viz extras first:  pip install gafro[viz]

Run with:
    python examples/franka_tsr_interactive.py
    python examples/franka_tsr_interactive.py --no-viz   # plan once, no scene
"""

from __future__ import annotations

import argparse
import time

import gafro as ga
import numpy as np
from tsr import TSR

from pycbirrt import CBiRRT, CBiRRTConfig
from pycbirrt.backends.gafro import GafroIKSolver, GafroRobotModel, as_motor

# Any robot description with a 7-DOF arm chain works; the default ships a panda
# whose "hand" chain is the arm (joints 1-7 ending at the hand frame).
DEFAULT_ROBOT = "/home/tobi/coding/src/rss/geodude.xml"
DEFAULT_CHAIN = "left_ur5e/endeffector_link"

# The goal TSR is centered on the end-effector pose at this reference config (a
# fraction of the way across the controlled joint range). Deriving the center
# from FK keeps the region reachable for whatever arm/chain is loaded, instead of
# hardcoding a world pose tuned for one robot.
TSR_REF_FRACTION = -0.1


class NoCollision:
    """Trivial collision checker: the scene has no obstacles for this demo."""

    def is_valid(self, q: np.ndarray) -> bool:
        return True


class TSRPlanner:
    """Holds the robot + CBiRRT and replans to a given TSR from a fixed start."""

    def __init__(self, seed: int = 1, robot_path: str = DEFAULT_ROBOT):
        self.seed = seed
        self.robot = GafroRobotModel.from_file(robot_path, chain_name=DEFAULT_CHAIN)
        self.ik = GafroIKSolver(self.robot.manipulator, self.robot.joint_limits,
                                base_configuration=self.robot.base_configuration,
                                max_iterations=300, tolerance=1e-6)
        self.config = CBiRRTConfig(
            max_iterations=5000, step_size=0.15, goal_bias=0.2, tsr_samples=50,
            angular_joints=(True,) * self.robot.dof,
        )
        self.planner = CBiRRT(self.robot, self.ik, NoCollision(), self.config)
        # Plan from the System's default configuration, extracted down to this
        # chain's controlled joints. The full default pose is kept as the base for
        # visualization so the rest of the robot renders correctly.
        self.start = self.robot.system_to_controlled(
            self.robot.default_system_configuration)

        # Reference goal pose derived from FK at a reachable config, so the TSR
        # region is reachable for whatever arm is loaded (see TSR_REF_FRACTION).
        ref_q = np.full(self.robot.dof, TSR_REF_FRACTION)
        T_ref = as_motor(self.robot.forward_kinematics(ref_q)).to_transformation_matrix()
        self.tsr_center = T_ref[:3, 3]
        self.tsr_rot = T_ref[:3, :3]

    def ee_xyz(self, q) -> np.ndarray:
        return as_motor(self.robot.forward_kinematics(q)).to_transformation_matrix()[:3, 3]

    def plan(self, tsr: TSR):
        """Plan from the fixed start into ``tsr``. Returns ``(path, info)``."""
        t0 = time.perf_counter()
        result = self.planner.plan(
            start=self.start, goal_tsrs=[tsr], seed=self.seed, return_details=True
        )
        dt = time.perf_counter() - t0
        if not result.success:
            return None, f"no path ({result.failure_reason})  [{dt:.2f}s]"
        d, _ = tsr.distance(self.robot.forward_kinematics(result.path[-1]))
        info = f"{len(result.path)} waypoints, goal dist {d:.3f}  [{dt:.2f}s]"
        return result.path, info


def make_tsr(center: np.ndarray, rot: np.ndarray,
             half_x: float, half_y: float, half_z: float, yaw: float) -> TSR:
    """Build the grasp TSR from a reference frame and slider values.

    ``center`` / ``rot`` are the TSR frame's world position and orientation;
    ``half_*`` are half-extents of the translation box (meters); ``yaw`` is the
    half-range of free rotation about the EE approach axis (radians), so
    ``yaw = pi`` is a full turn and ``yaw = 0`` pins the wrist orientation.
    """
    T0_w = np.eye(4)
    T0_w[:3, 3] = center
    T0_w[:3, :3] = rot
    Bw = np.array([
        [-yaw, yaw],         # b23 (free yaw about approach axis)
        [0.0, 0.0],          # b12 (tilt pinned)
        [0.0, 0.0],          # b13 (tilt pinned)
        [-half_x, half_x],   # tx
        [-half_y, half_y],   # ty
        [-half_z, half_z],   # tz
    ])
    return TSR(T0_w=T0_w, Tw_e=np.eye(4), Bw=Bw)


def visualize(planner: TSRPlanner, port: int):
    """Serve a viser scene: robot meshes, the TSR region, sliders, replanning."""
    system = planner.robot.system  # same System the planner does its math on

    viz = ga.Visualizer(port=port)
    robot_viz = viz.add_robot(system, joint_sliders=False)  # driven by playback

    # The TSR frame (drawn once) and the region box (resized on every replan),
    # both placed at the planner's reachable reference pose.
    viz.scene.add_frame("/tsr/frame", axes_length=0.12, axes_radius=0.005,
                        position=planner.tsr_center, wxyz=_quat(planner.tsr_rot))
    tsr_box = viz.scene.add_box("/tsr/box", color=(255, 170, 0), dimensions=(0.1, 0.1, 0.1),
                                opacity=0.25, position=planner.tsr_center)
    path_spline = {"node": None}
    ee_start = viz.add_point(ga.Point(*planner.ee_xyz(planner.start)),
                             name="/ee_start", color=(0, 80, 255), radius=0.025)
    ee_frame = viz.add_motor(planner.robot.forward_kinematics(planner.start), name="/ee_pose",
                             axes_length=0.12, axes_radius=0.005)
    robot_viz.update(planner.robot.to_system_configuration(planner.start))

    # -- GUI: TSR extent sliders + a status readout ------------------------
    with viz.gui.add_folder("TSR region (Bw)"):
        s_x = viz.gui.add_slider("half x", min=0.0, max=0.3, step=0.005, initial_value=0.1)
        s_y = viz.gui.add_slider("half y", min=0.0, max=0.3, step=0.005, initial_value=0.1)
        s_z = viz.gui.add_slider("half z", min=0.0, max=0.3, step=0.005, initial_value=0.0)
        s_yaw = viz.gui.add_slider("yaw range", min=0.0, max=np.pi, step=0.01,
                                   initial_value=np.pi)
        status = viz.gui.add_text("status", initial_value="-", disabled=True)

    state = {"dirty": True, "path": None, "frame": 0}

    def mark_dirty(_event=None):
        state["dirty"] = True

    for s in (s_x, s_y, s_z, s_yaw):
        s.on_update(mark_dirty)

    def replan():
        tsr = make_tsr(planner.tsr_center, planner.tsr_rot,
                       s_x.value, s_y.value, s_z.value, s_yaw.value)
        # Update the region's box: translation half-extents -> full dimensions
        # (a zero extent gets a thin sliver so it stays visible).
        dims = (max(2 * s_x.value, 0.01), max(2 * s_y.value, 0.01), max(2 * s_z.value, 0.01))
        tsr_box.dimensions = dims
        path, info = planner.plan(tsr)
        status.value = info
        print(f"  replan: {info}")
        if path is not None:
            state["path"], state["frame"] = path, 0
            ee = np.array([planner.ee_xyz(q) for q in path])
            if path_spline["node"] is not None:
                path_spline["node"].remove()
            path_spline["node"] = viz.scene.add_spline_catmull_rom(
                "/ee_path", positions=ee, color=(0, 0, 0), line_width=3.0)

    def tick():
        # Debounced replanning: replan once after the slider stream settles.
        if state["dirty"]:
            state["dirty"] = False
            replan()
            return
        # Animate the arm along the current path (loops).
        path = state["path"]
        if not path:
            return
        i = state["frame"] % len(path)
        q = path[i]
        robot_viz.update(planner.robot.to_system_configuration(q))
        ee_frame.gafro = planner.robot.forward_kinematics(q)
        ee_frame.redraw()
        state["frame"] = (state["frame"] + 1) % len(path)
        time.sleep(0.1)

    viz.add_ticker(tick)
    viz.show("\nviser running at http://localhost:%d  (drag the TSR sliders; ctrl-c to quit)"
             % port)


def _quat(rot3):
    from gafro.visualization import quaternion_from_matrix

    m = np.eye(4)
    m[:3, :3] = rot3
    return quaternion_from_matrix(m)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--port", type=int, default=8080, help="viser port (default: 8080)")
    parser.add_argument("--robot", default=DEFAULT_ROBOT,
                        help=f"Robot description with a 7-DOF arm chain (default: {DEFAULT_ROBOT})")
    parser.add_argument("--no-viz", action="store_true", help="Plan once; skip the viser scene")
    args = parser.parse_args()

    print("=" * 64)
    print("Interactive-TSR planning")
    print("=" * 64)

    planner = TSRPlanner(seed=args.seed, robot_path=args.robot)
    print(f"  {planner.robot.system.get_name()} ({planner.robot.dof} DOF) via the gafro CGA backend")

    if args.no_viz:
        tsr = make_tsr(planner.tsr_center, planner.tsr_rot, 0.1, 0.1, 0.0, np.pi)
        path, info = planner.plan(tsr)
        print(f"  Plan to default TSR: {info}")
    else:
        visualize(planner, args.port)


if __name__ == "__main__":
    main()
