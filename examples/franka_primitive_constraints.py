# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Example: Franka CBiRRT planning under CGA primitive path constraints, in viser.

Demonstrates the geometric-primitive constraints (``tsr.PlaneConstraint`` and
``tsr.SphereConstraint``) on the 7-DOF Franka Emika arm, driven entirely by the
gafro CGA backend -- no MuJoCo / EAIK required. Each constraint keeps the
end-effector *origin* on a CGA primitive for the whole trajectory:

* ``plane``  -- the EE stays on a tilted world plane (a flat slice of the
  workspace), the 3D analogue of the planar-arm line example.
* ``sphere`` -- the EE stays on a spherical shell about a center point (think
  wiping a dome / keeping a constant standoff from a point).

Both primitives use the native CGA ``Plane.project`` / ``Sphere.project``
formulas under the hood: the witness the planner projects onto is the foot of
that projection. Start and goal configurations are placed *exactly* on the
manifold by snapping a seed pose onto it and solving IK, so the endpoints
already satisfy the path constraint.

Two views of the same robot (the ``gafro.Visualizer`` idiom), both backed by
one ``gafro.System``:
  * ``robot`` (a ``GafroRobotModel`` over the arm chain) does the math -- FK,
    geometric Jacobian, joint limits -- and feeds the gafro CBiRRT backend.
  * ``robot.system`` carries the visual meshes the ``Visualizer`` drives with
    FK. Their EE poses coincide.

Install the viz extras first:  pip install gafro[viz]

Run with:
    python examples/franka_primitive_constraints.py                  # plane
    python examples/franka_primitive_constraints.py --constraint sphere
    python examples/franka_primitive_constraints.py --no-viz         # plan only
"""

from __future__ import annotations

import argparse
import time

import gafro as ga
import numpy as np
from tsr import PlaneConstraint, SphereConstraint

from pycbirrt import CBiRRT, CBiRRTConfig
from pycbirrt.backends.gafro import GafroIKSolver, GafroRobotModel, as_motor

# Any robot description with a 7-DOF arm chain works; the default ships a panda
# whose "hand" chain is the arm (joints 1-7 ending at the hand frame).
DEFAULT_ROBOT = "/home/tobi/coding/src/rss/panda_new.xml"
DEFAULT_CHAIN = "hand"


class NoCollision:
    """Trivial collision checker: the scene has no obstacles for this demo."""

    def is_valid(self, q: np.ndarray) -> bool:
        return True


def _ee_xyz(robot: GafroRobotModel, q: np.ndarray) -> np.ndarray:
    return as_motor(robot.forward_kinematics(q)).to_transformation_matrix()[:3, 3]


def snap_config_to_constraint(robot, ik, q_seed, constraint) -> np.ndarray:
    """Return a config near ``q_seed`` whose EE pose lies on the constraint.

    Projects the seed EE pose onto the constraint manifold (``distance`` returns
    the projected witness) and solves IK back to joint space from the seed, so
    the result is a genuine configuration sitting on the manifold.
    """
    seed_pose = robot.forward_kinematics(q_seed)
    _, witness = constraint.distance(seed_pose)
    target = constraint.to_transform(witness)
    sols = ik.solve(target, q_init=q_seed)
    if not sols:
        raise RuntimeError("IK failed to place a start/goal config on the constraint manifold")
    return sols[0]


def make_plane_constraint(robot, q_mid) -> PlaneConstraint:
    """A tilted plane through the workspace, passing through the seed EE position."""
    p = _ee_xyz(robot, q_mid)
    # Tilt the plane normal ~25 deg off vertical so the slice is visibly oblique.
    normal = [np.sin(np.radians(25)), 0.0, np.cos(np.radians(25))]
    return PlaneConstraint.from_point_normal(p, normal)


def make_sphere_constraint(robot, q_mid) -> SphereConstraint:
    """A spherical shell about a point below the base, through the seed EE."""
    center = np.array([0.0, 0.0, 0.2])
    radius = float(np.linalg.norm(_ee_xyz(robot, q_mid) - center))
    return SphereConstraint.from_center_radius(center, radius)


def plan(constraint_kind: str, seed: int, robot_path: str = DEFAULT_ROBOT):
    """Build the robot, the constraint, and a CBiRRT path on the manifold."""
    robot = GafroRobotModel.from_file(robot_path, chain_name=DEFAULT_CHAIN)
    ik = GafroIKSolver(robot.manipulator, robot.joint_limits, max_iterations=300, tolerance=1e-4,
                       base_configuration=robot.base_configuration)
    print(f"  {robot.system.get_name()} ({robot.dof} DOF) via the gafro CGA backend")

    # Two seed configurations whose EE positions straddle the workspace; we snap
    # each onto the manifold so the endpoints already satisfy the constraint.
    q_seed_a = np.array([0.6, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])
    q_seed_b = np.array([-0.6, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])
    q_mid = 0.5 * (q_seed_a + q_seed_b)

    if constraint_kind == "plane":
        constraint = make_plane_constraint(robot, q_mid)
    elif constraint_kind == "sphere":
        constraint = make_sphere_constraint(robot, q_mid)
    else:
        raise ValueError(f"Unknown constraint kind: {constraint_kind!r}")
    print(f"  Constraint: {constraint}")

    start = snap_config_to_constraint(robot, ik, q_seed_a, constraint)
    goal = snap_config_to_constraint(robot, ik, q_seed_b, constraint)
    for label, q in (("Start", start), ("Goal", goal)):
        off = constraint.distance(robot.forward_kinematics(q))[0]
        print(f"  {label} EE: {np.round(_ee_xyz(robot, q), 3)}  (off-manifold: {off:.4f})")

    config = CBiRRTConfig(
        max_iterations=5000,
        step_size=0.1,
        goal_bias=0.1,
        tsr_tolerance=0.001,  # allowed distance off the constraint manifold
        angular_joints=(True,) * robot.dof,
    )
    planner = CBiRRT(robot, ik, NoCollision(), config)

    print("\n  Planning along the constraint manifold...")
    t0 = time.perf_counter()
    result = planner.plan(
        start=start, goal=goal, constraint_tsrs=[constraint], seed=seed, return_details=True
    )
    dt = time.perf_counter() - t0

    if not result.success:
        raise RuntimeError(f"No path found ({result.failure_reason})")

    max_violation = max(constraint.distance(robot.forward_kinematics(q))[0] for q in result.path)
    print(f"  Found path: {len(result.path)} waypoints in {dt:.2f}s ({result.iterations} iters)")
    print(f"  Max distance off the constraint manifold: {max_violation:.4f}")
    return robot, constraint, result.path


def visualize(robot, constraint, path, constraint_kind, port: int):
    """Serve a viser scene: the robot meshes, the CGA primitive, and the path."""
    # The same System the model does its math on also carries the visual meshes
    # the Visualizer drives with FK -- no second load, EE poses coincide.
    system = robot.system

    viz = ga.Visualizer(port=port)
    franka = viz.add_robot(system, joint_sliders=False)  # config is driven by playback

    # The constraint primitive is interactive: drag/resize it to see where it sits.
    ee = np.array([_ee_xyz(robot, q) for q in path])
    if constraint_kind == "plane":
        viz.add_plane(constraint.plane, name="/constraint",
                      origin_point=ga.Point(*ee.mean(axis=0)), size=1.0, interactive=True)
    else:
        viz.add_sphere(constraint.sphere, name="/constraint", interactive=True)

    # End-effector trace: spline + start/goal markers + a live EE frame.
    viz.add_spline(ee, name="/ee_path")
    viz.add_point(ga.Point(*ee[0]), name="/ee_start", color=(0, 80, 255), radius=0.025)
    viz.add_point(ga.Point(*ee[-1]), name="/ee_goal", color=(0, 200, 80), radius=0.025)
    ee_frame = viz.add_motor(robot.forward_kinematics(path[0]), name="/ee_pose",
                             axes_length=0.12, axes_radius=0.005)

    def on_frame(_i, q):
        q_full = np.zeros(system.get_dof())
        q_full[: robot.dof] = np.asarray(q, dtype=float)
        franka.update(q_full)
        ee_frame.gafro = robot.forward_kinematics(q)
        ee_frame.redraw()

    viz.add_playback(path, on_frame)
    viz.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraint", choices=["plane", "sphere"], default="plane",
                        help="Which primitive constraint to keep the EE on (default: plane)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--port", type=int, default=8080, help="viser port (default: 8080)")
    parser.add_argument("--robot", default=DEFAULT_ROBOT,
                        help=f"Robot description with a 7-DOF arm chain (default: {DEFAULT_ROBOT})")
    parser.add_argument("--no-viz", action="store_true", help="Plan only; skip the viser scene")
    args = parser.parse_args()

    print("=" * 64)
    print(f"Primitive-constraint planning: {args.constraint}")
    print("=" * 64)

    robot, constraint, path = plan(args.constraint, args.seed, args.robot)

    if args.no_viz:
        print("\nSkipping visualization (--no-viz)")
    else:
        visualize(robot, constraint, path, args.constraint, args.port)


if __name__ == "__main__":
    main()
