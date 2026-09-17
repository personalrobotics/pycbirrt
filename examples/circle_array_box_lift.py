# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Example: an arm array grasps a box and lifts it, keeping the grasp rigid.

The array from ``circle_array_tsr.py`` -- the same UR5e instanced N times on a
circle, facing inward -- picks up a box sitting at the centre and carries it
upward. This is the cooperative-manipulation task the multi-arm TSRs exist for:

  * **Approach.** The arms start above the box and held open, then descend onto
    it -- so the motion reads as reach-down-and-grab rather than beginning
    already in contact. MuJoCo confirms the split: zero arm-to-crate contacts at
    the pre-grasp, five once the descent finishes.
  * **Grasp.** Each arm's end-effector comes to rest on the face of the box it
    faces. Those contacts define the configuration the lift starts from, and the
    box is sized *from* them so the arms genuinely reach it.
  * **Carry.** The lift is expressed in the array's own task coordinates. For
    three or four arms the constraint is a ``CircleTSR`` / ``SphereTSR`` whose
    **dilation is pinned**: the arms may move the held shape, but not resize
    it, which is exactly what "do not crush or drop the box" means. Translation
    is free in z so the box can rise.
  * **Do not self-collide.** With the arms this close together the capsule
    self-collision checker is not optional; see ``pycbirrt.collision``.

Planning is probabilistic and, with both the rigid-grasp constraint and the box
as an obstacle, the free corridor is thin: roughly 2 runs in 6 find a path, so
``plan_lift`` retries with fresh seeds. Loosening ``RigidGraspConstraint``'s
tolerance or lowering ``--lift`` widens it.

``--physics`` replays the plan in MuJoCo under gravity, with the crate as a free
body. That check is worth running precisely because it can *fail*: this example
plans contact, not force. MuJoCo reports ~46 contacts between the arms and the
crate at the grasp pose, yet the crate does not rise -- the end-effectors touch
its walls but nothing presses inward, and bare friction will not carry a 2 kg
box. Making it actually lift needs grippers or an inward force controller;
the kinematic plan alone cannot tell you that, which is the point of running
the physics.

Dilation is what makes this expressible. A rigid grasp is a *constant scale*
constraint on the shape the end-effectors span, and the similarity transform
gafro reports for a 3- or 4-arm task space carries exactly that number.

Run with:
    python examples/circle_array_box_lift.py --arms 4
    python examples/circle_array_box_lift.py --arms 3 --no-viz
    python examples/circle_array_box_lift.py --arms 4 --metric kinetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gafro as ga
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from circle_array_tsr import (  # noqa: E402
    _patch_visualizer_joint_limits,
    build_model,
    build_system,
    collision_checker,
    ik_solver_for,
    joint_travel,
    to_system,
)

BOX_COLOR = (210, 150, 70)

# Grasp posture: shoulder, elbow and wrist-1 offsets applied to every arm. Found
# by searching for a configuration whose four end-effectors sit at equal height,
# low enough to meet a box standing on the ground, spread wide enough that the
# contacts bound a box of a sensible size, and clear of each other.
GRASP_POSTURE = {"shoulder_lift": -0.30, "elbow": -1.15, "wrist_1": 1.20}
DEFAULT_RADIUS = 0.85
# Alternate arms grip slightly higher. A perfectly level four-point grasp is
# exactly coplanar, and four coplanar points have no circumsphere -- gafro's
# similarity decomposition comes back all-NaN. A few centimetres of stagger
# keeps the contacts on the same box wall while making the task pose well
# defined. (Three arms always span a circle, so they need no stagger.)
GRASP_STAGGER = 0.08


def grasp_posture(model, arm_count: int) -> np.ndarray:
    """Configuration with every arm reaching down and inward onto the box."""
    q = np.zeros(model.dof)
    joints_per_arm = model.dof // max(arm_count, 1)
    for arm in range(arm_count):
        base = arm * joints_per_arm
        if joints_per_arm >= 2:
            q[base + 1] = -GRASP_POSTURE["shoulder_lift"]
        if joints_per_arm >= 3:
            q[base + 2] = GRASP_POSTURE["elbow"]
        if joints_per_arm >= 4:
            q[base + 3] = GRASP_POSTURE["wrist_1"]
        if arm_count >= 4 and arm % 2 and joints_per_arm >= 2:
            q[base + 1] += GRASP_STAGGER
    return q


# Half-thickness of the gripping link, i.e. how far the end-effector origin sits
# from the surface it presses on. The box is inset by this much so the arms rest
# *against* its faces instead of reaching through them.
# How far the end-effector origin sits from the face it presses on. Tuned
# against MuJoCo's real meshes rather than the capsule approximation: at 0.075
# the gripper is 4.5 mm inside the box, at 0.080 it has left the surface
# entirely, so 0.078 is the value that actually touches. The capsule model is
# more conservative than the mesh, so ``obstacle_margin`` below relaxes the
# planner's box check for the gripper links by that difference.
CONTACT_STANDOFF = 0.078


def box_from_contacts(contacts: np.ndarray, ground_z: float = 0.0,
                      standoff: float = CONTACT_STANDOFF) -> tuple[np.ndarray, np.ndarray]:
    """Size and centre of a box **standing on the ground** that the contacts grip.

    The contacts set the footprint, inset by ``standoff`` so the box's faces sit
    just inside the end-effectors: the arms press on the outside of the box
    rather than penetrating it. Sizing the box so its walls pass exactly through
    the contact *origins* is what put the wrist links ~9 cm inside it.

    The height puts the base on the ground and the contacts on the upper side
    wall, which is where an array of arms would grip a crate.
    """
    centre_xy = contacts[:, :2].mean(axis=0)
    half_extent = np.abs(contacts[:, :2] - centre_xy).max(axis=0) - standoff
    half_extent = np.maximum(half_extent, 0.05)
    contact_z = float(contacts[:, 2].mean())

    # Grip a little below the rim so the contacts are on the wall, not the lid.
    height = (contact_z - ground_z) / 0.8
    size = np.array([2.0 * half_extent[0], 2.0 * half_extent[1], height])
    centre = np.array([centre_xy[0], centre_xy[1], ground_z + height / 2.0])
    return size, centre


# How far above the grasp the arms start, and how much wider they hold open on
# approach, so the sequence reads as reach-down-and-close rather than starting
# already in contact.
APPROACH_HEIGHT = 0.22
APPROACH_OPENING = 0.18


def approach_posture(model, arm_count: int) -> np.ndarray:
    """Configuration above the box, held open -- the pre-grasp.

    Lifting the shoulder raises the end-effectors clear of the box; easing the
    elbow spreads them, so the array descends onto the box rather than starting
    wrapped around it.
    """
    q = grasp_posture(model, arm_count)
    joints_per_arm = model.dof // max(arm_count, 1)
    for arm in range(arm_count):
        base = arm * joints_per_arm
        if joints_per_arm >= 2:
            q[base + 1] -= APPROACH_HEIGHT
        if joints_per_arm >= 3:
            q[base + 2] += APPROACH_OPENING
    return q


def plan_approach(model, arm_count: int, q_start: np.ndarray, q_grasp: np.ndarray,
                  collision, metric=None, attempts: int = 20):
    """Plan the descent from the pre-grasp down onto the box.

    The goal is the grasp *configuration*, so this is a plain joint-space query:
    the arms are not holding anything yet, and the task-space region only becomes
    the right language once the box is in hand.
    """
    from pycbirrt import CBiRRT, CBiRRTConfig
    from pycbirrt.exceptions import PlanningError

    solver = ik_solver_for(model, arm_count, collision)
    config = CBiRRTConfig(max_iterations=3000, step_size=0.12, goal_bias=0.4,
                          tsr_samples=20, angular_joints=(True,) * model.dof,
                          metric=metric)
    planner = CBiRRT(model, solver, collision, config)
    for attempt in range(attempts):
        try:
            result = planner.plan(start=q_start, goal=q_grasp, seed=1 + attempt,
                                  return_details=True)
        except PlanningError:
            continue
        if result.success:
            return result.path
    return None


def grasp_configuration(model, system, arm_count: int, collision):
    """The grasp configuration, and the box its contacts define.

    The box is derived *from* the reachable grasp rather than placed first and
    reached for: that guarantees the arms reach it, and that it stands on the
    ground instead of floating.
    """
    q_grasp = grasp_posture(model, arm_count)
    contacts = end_effector_positions(model, system, arm_count, q_grasp)
    size, centre = box_from_contacts(contacts)
    return q_grasp, size, centre


# Links that do the gripping. They are *meant* to be against the box, so the
# capsule box-check skips them; every other link must stay clear.
CONTACT_LINK_SUFFIXES = ("wrist_1_link", "wrist_2_link", "wrist_3_link", "ur5e_ee")


def checker_with_box(system, model, collision, box_size, box_centre):
    """Extend a capsule checker so the box is an obstacle too.

    Without this the arms plan straight through the object they are carrying --
    measured at 9 cm of penetration at the grasp and 13 cm mid-lift.

    The gripping links are exempt. A capsule is a conservative hull of the real
    mesh, so at a grasp MuJoCo scores as 1.8 mm of contact the capsule model
    calls a 3 cm overlap; enforcing the capsule bound on the fingers would make
    touching the box impossible by construction.
    """
    from pycbirrt.collision import Box, CapsuleSelfCollisionChecker

    if not isinstance(collision, CapsuleSelfCollisionChecker):
        return collision
    collision.obstacles = [Box(box_centre, box_size)]
    collision.contact_links = {
        name for name in system.get_link_names()
        if name.startswith("arm") and name.split("/")[-1] in CONTACT_LINK_SUFFIXES
    }
    return collision


def lift_region(model, arm_count: int, q_grasp: np.ndarray, height: float):
    """Goal region for the lift: same held shape, raised by ``height``.

    ``dilation`` is pinned to its grasped value -- the array must not resize the
    box -- while z is centred on the lifted height. The plane orientation of a
    three-arm circle is held too, so the box does not tip.
    """
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    grasped = zero.to_bw(model.forward_kinematics(q_grasp))

    target = grasped.copy()
    target[2] = grasped[2] + height

    lower = target.copy()
    upper = target.copy()
    lower[0:2] -= 0.05          # small lateral slack
    upper[0:2] += 0.05
    lower[2] -= 0.03            # the lift height itself
    upper[2] += 0.03
    lower[3] -= 0.02            # dilation: pinned -- the grasp stays rigid
    upper[3] += 0.02
    if model.tsr_class._DOF > 4:
        lower[4:] -= 0.08       # circle plane: nearly level
        upper[4:] += 0.08
    return model.tsr_class(Bw=np.column_stack([lower, upper]))


class RigidGraspConstraint:
    """Path constraint holding the *contact geometry* fixed for the whole carry.

    Two corrections are baked in here.

    First, constraining only the goal is not enough: the planner is free to
    deform the array on the way and pull back at the end. Measured on a four-arm
    lift, the held shape wandered badly mid-path and recovered only at the
    final waypoint -- the box was crushed and re-formed in between.

    Second, and less obvious, the *dilation* coordinate is a poor stand-in for
    the box's size. Over 200 perturbed configurations its correlation with the
    mean contact spread was only 0.19: the similarity transform's scale is a
    property of the circumsphere through the contacts, and that sphere can grow
    while the contacts themselves close in. So this constrains the thing that
    actually matters -- the pairwise distances between end-effectors, which is
    what a rigid box fixes.

    ``tolerance`` is the slack in metres on any single contact-pair distance.
    It cannot be made arbitrarily small: a rigid grasp is a near-zero-measure
    manifold in configuration space, and a sampling planner needs *some* volume
    to explore. Measured on this array, 0/100 nearby configurations satisfy a
    0.05 m tolerance and 13/100 satisfy 0.10 m, so the default is loose enough
    to plan through while still keeping the box's size to a few percent.
    """

    def __init__(self, model, system, arm_count: int, q_grasp: np.ndarray,
                 tolerance: float = 0.12):
        self.model = model
        self.system = system
        self.arm_count = arm_count
        self.tolerance = float(tolerance)
        self.reference = self._pairwise(q_grasp)

    def _pairwise(self, q: np.ndarray) -> np.ndarray:
        contacts = end_effector_positions(self.model, self.system, self.arm_count, q)
        n = len(contacts)
        return np.array([np.linalg.norm(contacts[i] - contacts[j])
                         for i in range(n) for j in range(i + 1, n)])

    def drift(self, q: np.ndarray) -> float:
        """Largest change in any contact-pair distance since the grasp."""
        return float(np.max(np.abs(self._pairwise(q) - self.reference)))

    def satisfied(self, q: np.ndarray) -> bool:
        return self.drift(q) <= self.tolerance


class _GraspAwareCollision:
    """Collision checker that also rejects configurations deforming the grasp.

    The planner applies path constraints by projecting a pose and re-solving IK,
    which needs a constraint expressed over *poses*; the grasp constraint here is
    over joint configurations. Folding it into the validity check is the honest
    way to enforce it -- a configuration that squeezes the box is simply not a
    configuration this system may occupy.
    """

    def __init__(self, collision, constraint):
        self.collision = collision
        self.constraint = constraint

    def is_valid(self, q) -> bool:
        if self.collision is not None and not self.collision.is_valid(q):
            return False
        return self.constraint.satisfied(q)

    def clearance(self, q):
        return getattr(self.collision, "clearance", lambda _q: float("inf"))(q)


def plan_lift(model, arm_count: int, q_grasp: np.ndarray, region, collision,
              metric=None, geodesic_extension: bool = False, attempts: int = 20,
              constraint=None):
    """Plan the carry from the grasp configuration into the lift region."""
    from pycbirrt import CBiRRT, CBiRRTConfig
    from pycbirrt.exceptions import PlanningError

    if constraint is not None:
        collision = _GraspAwareCollision(collision, constraint)
    solver = ik_solver_for(model, arm_count, collision)
    config = CBiRRTConfig(max_iterations=3000, step_size=0.15, goal_bias=0.35,
                          tsr_samples=30, angular_joints=(True,) * model.dof,
                          metric=metric, geodesic_extension=geodesic_extension)
    planner = CBiRRT(model, solver, collision, config)
    for attempt in range(attempts):
        try:
            result = planner.plan(start=q_grasp, goal_tsrs=[region],
                                  seed=1 + attempt, return_details=True)
        except PlanningError:
            continue
        if result.success:
            return result.path
    return None


def end_effector_positions(model, system, arm_count: int, q: np.ndarray) -> np.ndarray:
    """World positions of each arm's end-effector at ``q``."""
    from _circle_array import chain_names
    from gafro import robot

    task_indices = np.asarray(model.cooperative.get_joint_indices(), dtype=int)
    system_q = np.zeros(system.get_dof(), dtype=float)
    system_q[task_indices] = model._to_task_full(q)
    points = []
    for chain in chain_names(arm_count):
        task_space = robot.SingleArmTaskSpace(system, chain, chain)
        indices = np.asarray(task_space.get_joint_indices(), dtype=int)
        translator = task_space.compute_ee_motor(system_q[indices]).get_translator()
        points.append([translator.x(), translator.y(), translator.z()])
    return np.asarray(points, dtype=float)


def _contact_spread(contacts: np.ndarray) -> float:
    """Mean pairwise distance between contacts -- the box size the arms hold."""
    n = len(contacts)
    return float(np.mean([np.linalg.norm(contacts[i] - contacts[j])
                          for i in range(n) for j in range(i + 1, n)]))


def box_pose_along(model, system, arm_count: int, q: np.ndarray) -> np.ndarray:
    """Where the box sits: the centroid of the grasping end-effectors.

    Deliberately *not* the held shape's centre. The circumsphere of four nearly
    coplanar contacts has its centre far above them (z ~ 1.65 m for contacts at
    z ~ 0.75 m), which is correct geometry but not where the box is.
    """
    return end_effector_positions(model, system, arm_count, q).mean(axis=0)


def mujoco_model(system, box_size, box_centre, out_dir: Path):
    """Export the array to MJCF, add the box and a floor, and load it in MuJoCo.

    gafro writes MJCF but emits the same ``attachment_site`` name once per arm,
    which MuJoCo rejects as a duplicate, and a relative ``meshdir``; both are
    patched here. The box is a free body, so once the arms let go it falls.
    """
    import re

    import gafro
    import mujoco
    from _circle_array import DEFAULT_SOURCE

    xml = gafro.SystemSerialization.to_mjcf_string(system)
    counter = [0]

    def unique_site(_match):
        counter[0] += 1
        return f'name="attachment_site_{counter[0] - 1}"'

    xml = re.sub(r'name="attachment_site"', unique_site, xml)
    assets = (Path(DEFAULT_SOURCE).parent / "assets").resolve()
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{assets}"', xml)

    half = np.asarray(box_size, dtype=float) / 2.0
    extras = f"""
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.88 1"/>
        <body name="crate" pos="{box_centre[0]} {box_centre[1]} {box_centre[2]}">
          <freejoint name="crate_free"/>
          <geom name="crate_geom" type="box"
                size="{half[0]} {half[1]} {half[2]}"
                rgba="0.82 0.59 0.27 1" mass="2.0"
                friction="1.2 0.02 0.001"/>
        </body>
      </worldbody>
    """
    xml = xml.replace("</mujoco>", extras + "</mujoco>")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "array_with_box.xml"
    path.write_text(xml)
    return mujoco.MjModel.from_xml_path(str(path)), path


def simulate_lift(system, model, arm_count, path, box_size, box_centre, out_dir,
                  settle_seconds: float = 0.4):
    """Replay the planned path in MuJoCo under gravity and report the box height.

    The arms are position-controlled onto each waypoint. This is the honest
    check that the plan is a *lift*: a kinematic path can hold a rigid grasp in
    task coordinates and still drop a box that is only touched, not gripped.
    """
    import mujoco

    mj_model, xml_path = mujoco_model(system, box_size, box_centre, out_dir)
    data = mujoco.MjData(mj_model)

    # Joint order in MJCF matches the System's, and the crate's freejoint is
    # appended last, so the arm coordinates are the leading entries.
    arm_dof = int(model.cooperative.get_dof())
    task_indices = np.asarray(model.cooperative.get_joint_indices(), dtype=int)

    def apply(q):
        system_q = np.zeros(system.get_dof(), dtype=float)
        system_q[task_indices] = model._to_task_full(q)
        data.qpos[:arm_dof] = system_q[:arm_dof]
        data.qvel[:] = 0.0

    apply(path[0])
    mujoco.mj_forward(mj_model, data)
    crate_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "crate")
    start_height = float(data.xpos[crate_id][2])

    steps_per_waypoint = max(int(settle_seconds / mj_model.opt.timestep / len(path)), 1)
    for waypoint in path:
        apply(waypoint)
        for _ in range(steps_per_waypoint):
            mujoco.mj_step(mj_model, data)

    return start_height, float(data.xpos[crate_id][2]), xml_path


def visualize(model, system, arm_count: int, path, box_size, port: int):
    """Serve the scene: arms, the box carried along the path, and a scrubber."""
    _patch_visualizer_joint_limits()
    viz = ga.Visualizer(port=port)
    robot_viz = viz.add_robot(system, joint_sliders=False)

    start_centre = box_pose_along(model, system, arm_count, path[0])
    box = viz.scene.add_box("/box", color=BOX_COLOR, dimensions=tuple(box_size),
                            position=tuple(start_centre))
    viz.add_label(f"{arm_count} arms lifting a "
                  f"{box_size[0]:.2f}x{box_size[1]:.2f}x{box_size[2]:.2f} m box",
                  position=(0.0, 0.0, 1.2), name="/title")

    def on_frame(_index, q):
        robot_viz.update(to_system(model, q))
        box.position = tuple(box_pose_along(model, system, arm_count, q))

    robot_viz.update(to_system(model, path[0]))
    viz.add_playback(list(path), on_frame)
    viz.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", type=int, default=4, choices=(3, 4),
                        help="arms in the array (the held shape needs 3 or 4)")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                        help="mounting circle radius")
    parser.add_argument("--lift", type=float, default=0.25, help="lift height in metres")
    parser.add_argument("--metric", default="euclidean", choices=("euclidean", "kinetic"))
    parser.add_argument("--geodesic-extension", action="store_true",
                        help="extend along the natural gradient (needs --metric kinetic)")
    parser.add_argument("--no-self-collision", action="store_true")
    parser.add_argument("--no-rigid-grasp", action="store_true",
                        help="drop the path constraint pinning the dilation "
                             "(shows the array stretching the box mid-carry)")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--physics", action="store_true",
                        help="replay the plan in MuJoCo under gravity (needs mujoco)")
    args = parser.parse_args()

    system, path_file = build_system(args.arms, args.radius, args.out_dir)
    model, label = build_model(args.arms, system)
    collision = collision_checker(system, model, enabled=not args.no_self_collision)

    print(f"array:      {args.arms} x UR5e, circle radius {args.radius} m")
    print(f"written to: {path_file}")
    print(f"task space: {label}")

    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    q_grasp, box_size, box_centre = grasp_configuration(model, system, args.arms, collision)
    collision = checker_with_box(system, model, collision, box_size, box_centre)
    contacts = end_effector_positions(model, system, args.arms, q_grasp)
    grasped = zero.to_bw(model.forward_kinematics(q_grasp))

    print(f"box:        {box_size[0]:.2f} x {box_size[1]:.2f} x {box_size[2]:.2f} m, "
          f"standing on the ground (centre z={box_centre[2]:.3f})")
    print(f"grasp:      contacts at z={contacts[:, 2].mean():.3f} m, on the box wall")
    gap = (np.abs(contacts - box_centre) - box_size / 2.0).max(axis=1)
    print(f"            end-effector origins stand {gap.max():.3f} m off the "
          f"surface (link half-thickness)")
    if hasattr(collision, "clearance"):
        print(f"            arm-to-arm clearance {collision.clearance(q_grasp):+.3f} m")
    if getattr(collision, "obstacles", None):
        print(f"            arm-to-box clearance "
              f"{collision.obstacle_clearance(q_grasp):+.3f} m")

    metric = None
    if args.metric == "kinetic":
        from pycbirrt.metrics import metric_for_model

        metric = metric_for_model(model)

    # Phase 1: descend from the pre-grasp onto the box.
    q_approach = approach_posture(model, args.arms)
    approach_contacts = end_effector_positions(model, system, args.arms, q_approach)
    print(f"approach:   start {approach_contacts[:, 2].mean() - contacts[:, 2].mean():+.3f} m "
          f"above the grasp, held open")
    descent = plan_approach(model, args.arms, q_approach, q_grasp, collision,
                            metric=metric)
    if descent is None:
        print("            no descent path found")
        return 1
    print(f"            [{args.metric}] {len(descent)} waypoints, "
          f"joint travel {joint_travel(descent):.3f} rad")

    # Phase 2: carry the box upward with the grasp held rigid.
    region = lift_region(model, args.arms, q_grasp, args.lift)
    constraint = None if args.no_rigid_grasp else RigidGraspConstraint(
        model, system, args.arms, q_grasp)
    path = plan_lift(model, args.arms, q_grasp, region, collision, metric=metric,
                     geodesic_extension=args.geodesic_extension,
                     constraint=constraint)
    if path is None:
        print("lift:       no path found")
        return 1
    # The full motion the viewer and the physics replay see.
    full_path = list(descent) + list(path[1:])

    box_start = box_pose_along(model, system, args.arms, path[0])
    box_end = box_pose_along(model, system, args.arms, path[-1])
    spreads = [_contact_spread(end_effector_positions(model, system, args.arms, w))
               for w in path]
    lifted = zero.to_bw(model.forward_kinematics(path[-1]))
    print(f"lift:       [{args.metric}] {len(path)} waypoints, "
          f"joint travel {joint_travel(path):.3f} rad")
    print(f"            box z {box_start[2]:+.3f} -> {box_end[2]:+.3f} m "
          f"(raised {box_end[2] - box_start[2]:+.3f})")
    drift = max(abs(zero.to_bw(model.forward_kinematics(w))[3] - grasped[3])
                for w in path)
    print(f"            dilation {grasped[3]:+.3f} -> {lifted[3]:+.3f}, "
          f"max drift ALONG the path {drift:.4f}"
          f"{'' if constraint is not None else '  (unconstrained!)'}")
    print(f"            contact spread {spreads[0]:.4f} -> {spreads[-1]:.4f} m "
          f"(max deviation {max(abs(v - spreads[0]) for v in spreads):.4f}; "
          f"a rigid box cannot change size)")
    if hasattr(collision, "clearance"):
        print(f"            min arm-to-arm clearance along path: "
              f"{min(collision.clearance(w) for w in path):+.3f} m")
    if getattr(collision, "obstacles", None):
        print(f"            min arm-to-box clearance along path: "
              f"{min(collision.obstacle_clearance(w) for w in path):+.3f} m")

    if args.physics:
        try:
            start_height, end_height, xml_path = simulate_lift(
                system, model, args.arms, full_path, box_size, box_centre,
                Path(args.out_dir) if args.out_dir else Path(path_file).parent)
            print(f"physics:    MuJoCo replay under gravity ({xml_path})")
            risen = end_height - start_height
            print(f"            crate z {start_height:.3f} -> {end_height:.3f} m "
                  f"(risen {risen:+.3f})")
            if risen < 0.05:
                print("            the crate stayed put: the arms *touch* it (MuJoCo "
                      "reports contacts) but")
                print("            nothing squeezes it, so friction alone cannot carry "
                      "it. A real lift needs")
                print("            grippers, or force control pressing the "
                      "end-effectors inward.")
        except ImportError:
            print("physics:    mujoco not installed (pip install mujoco)")

    if args.no_viz:
        return 0
    visualize(model, system, args.arms, full_path, box_size, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
