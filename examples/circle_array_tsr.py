# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Example: one manipulator, arrayed 1-4 times on a circle, under a TSR.

The *same* UR5e is instanced N times, evenly spaced on a circle and all facing
inward, so their workspaces overlap near the centre. The point is to show how
the task-space description -- and therefore the TSR -- changes with the number
of cooperating arms, on an otherwise identical robot:

    arms  task space                      TSR          constrained coordinates
    ----  -----------------------------   ----------   ----------------------------
    1     SingleArmTaskSpace              TSR          [b12,b13,b23, tx,ty,tz]   (6)
    2     DualArmTaskSpace                BimanualTSR  absolute + relative Motor (6+6)
    3     TripleCooperativeTaskSpace      CircleTSR    [tx,ty,tz, dilation, n1,n2] (6)
    4     QuadrupleCooperativeTaskSpace   SphereTSR    [tx,ty,tz, dilation]        (4)

Three and four arms span a *circle* and a *sphere*, and gafro describes them by
a ``SimilarityTransformation`` -- which carries a **dilation** as well as a
rotation and translation, because the arms can grow or shrink the shape they
hold. The symmetry of the spanned primitive is why not every rotation is
constrained: a sphere is unchanged by any rotation about its centre (so
``SphereTSR`` has none), and a circle is unchanged by spin about its own axis
(so ``CircleTSR`` keeps only the two that tilt its plane).

The visualization shows the arms, the spanned primitive, and -- for 3 and 4
arms -- a dilation slider that resizes the held shape and re-solves IK, which
is the clearest way to see the extra coordinate doing real work.

Install the viz extras first:  pip install gafro[viz]

Planning (``--plan``) has two tasks:

    ``--task move`` (default)  plan from the seed configuration to a goal region
                               of the right kind for this arm count -- a TSR, a
                               BimanualTSR, a CircleTSR or a SphereTSR. Works
                               for every arm count.
    ``--task grow``            enlarge the held circle / sphere (3 and 4 arms).

Known limitation, two arms: the bimanual IK has to satisfy the absolute *and*
relative pose at once, and from a cold start it often fails on poses that are
demonstrably reachable -- it solves them when seeded near the answer, and a
zero-width goal region reproducing the target pose to 1e-15 still does not
solve unseeded. The planner seeds its goal tree through exactly that cold-start
IK, so the two-arm ``move`` task frequently finds no path however many times it
is retried. This is a property of ``GafroBimanualIKSolver``, not of the goal
construction or the collision checker; the other arm counts are unaffected.
(``--task grow`` needs a held shape, so it is 3-4 arms only -- there is no
working ``--plan`` for two arms at present.) ``--geodesic-extension`` also does
not connect for two arms.

Run with:
    python examples/circle_array_tsr.py --arms 4
    python examples/circle_array_tsr.py --arms 3 --no-viz
    python examples/circle_array_tsr.py --arms 1 --no-viz
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import gafro as ga
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _circle_array import COOPERATIVE_TASK_SPACE, chain_names, write_circle_array  # noqa: E402

# Per-arm-count colours for the spanned primitive.
PRIMITIVE_COLOR = (255, 170, 0)


def _patch_visualizer_joint_limits() -> None:
    """Work around a gafro Visualizer bug when reading System joint limits.

    ``gafro.visualization.robot.Robot._actuated_joints`` does
    ``np.asarray(system.get_joint_limits_min())``,
    but that returns a ``JointPosition`` wrapper rather than an array, so numpy
    produces a 0-d array and the subsequent ``lo[idx]`` raises IndexError. This
    affects *every* robot, not just the composed arrays here. Patch the accessor
    to go through ``coefficients()`` when present.
    """
    try:
        from gafro.visualization import robot as robot_viz
    except ImportError:  # pragma: no cover - viz extras not installed
        return
    visual = getattr(robot_viz, "Robot", None)
    original = getattr(visual, "_actuated_joints", None)
    if visual is None or original is None or getattr(original, "_limits_patched", False):
        return

    def _vector(value):
        coefficients = getattr(value, "coefficients", None)
        if callable(coefficients):
            value = coefficients()
        return np.asarray(value, dtype=float).ravel()

    def _actuated_joints(self):
        low = _vector(self.system.get_joint_limits_min())
        high = _vector(self.system.get_joint_limits_max())
        joints = []
        for name in self.system.get_joint_names():
            joint = self.system.get_joint(name)
            if not joint.is_actuated():
                continue
            index = joint.get_index()
            joints.append((name, index, float(low[index]), float(high[index])))
        joints.sort(key=lambda entry: entry[1])
        return joints

    _actuated_joints._limits_patched = True
    visual._actuated_joints = _actuated_joints


def build_system(arm_count: int, radius: float, out_dir: Path | None = None):
    """Compose and load the N-arm circular array."""
    directory = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="circle_array_"))
    path = directory / f"circle_array_{arm_count}.yaml"
    write_circle_array(arm_count, path, radius=radius)
    return ga.SystemSerialization.load(str(path)), path


def build_model(arm_count: int, system):
    """The pycbirrt model matching this arm count, plus a label for it."""
    chains = chain_names(arm_count)
    if arm_count == 1:
        from pycbirrt.backends.gafro import GafroRobotModel

        return GafroRobotModel(system, chain_name=chains[0]), "TSR (single arm)"
    if arm_count == 2:
        from pycbirrt.backends.gafro_bimanual import GafroBimanualModel

        return (GafroBimanualModel(system, COOPERATIVE_TASK_SPACE),
                "BimanualTSR (absolute + relative)")
    from pycbirrt.backends.gafro_multiarm import GafroMultiArmModel

    model = GafroMultiArmModel(system, chains)
    return model, f"{model.tsr_class.__name__} ({model.tsr_class._DOF} DOF, incl. dilation)"


# Alternate arms are lifted by this much to break the array's symmetry.
SEED_STAGGER = 0.25
# Seed posture: shoulder lifted and elbow tucked so the arms reach inward with
# bent elbows. Stretched-flat arms foul each other on a small circle; tucked
# ones keep ~0.33 m of capsule clearance even at radius 0.7.
SEED_SHOULDER_LIFT = 0.6
SEED_ELBOW = -1.6


def seed_configuration(model, arm_count: int) -> np.ndarray:
    """A configuration with every arm reaching inward toward the circle centre.

    Alternating arms are staggered in shoulder-lift. This matters for the
    four-arm case: with every arm at an identical angle the array is perfectly
    symmetric and the four end-effectors come out **exactly coplanar**, and four
    coplanar points do not define a finite sphere -- the spanned sphere degenerates
    (its radius runs off to ~1e9 and the dilation coordinate diverges, since
    ``atanh`` blows up as the dilator ratio approaches 1). Breaking the symmetry
    gives the four points a real circumsphere and makes the dilation coordinate
    well-conditioned.

    Three arms never have this problem: any three non-collinear points define a
    circle. The stagger is applied anyway to keep the arm counts comparable.
    """
    q = np.zeros(model.dof)
    joints_per_arm = model.dof // max(arm_count, 1)
    for arm in range(arm_count):
        base = arm * joints_per_arm
        if joints_per_arm >= 2:
            q[base + 1] = -SEED_SHOULDER_LIFT
        if joints_per_arm >= 3:
            q[base + 2] = SEED_ELBOW
        # Stagger alternate arms so the four end-effectors are not coplanar.
        if arm % 2 and joints_per_arm >= 2:
            q[base + 1] += SEED_STAGGER
    return q


def describe_pose(model, arm_count: int, q: np.ndarray) -> str:
    """One-line readout of the task-space pose for this arm count."""
    pose = model.forward_kinematics(q)
    if arm_count == 1:
        translator = pose.get_translator()
        return f"EE position [{translator.x():+.3f} {translator.y():+.3f} {translator.z():+.3f}]"
    if arm_count == 2:
        absolute = pose.absolute.get_translator()
        return (f"absolute [{absolute.x():+.3f} {absolute.y():+.3f} {absolute.z():+.3f}]"
                f"  relative |log| {np.linalg.norm(np.asarray(pose.relative.log())):.3f}")
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    coordinates = zero.to_bw(pose)
    labels = model.tsr_class._LABELS
    return "  ".join(f"{name}={value:+.3f}" for name, value in zip(labels, coordinates))


def plan_region(model, arm_count: int, q_seed: np.ndarray, half_width: float = 0.05):
    """A TSR centred on the seed pose, of the shape this arm count calls for."""
    if arm_count in (1, 2):
        return None  # single/dual regions are built by their own demos
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    centre = zero.to_bw(model.forward_kinematics(q_seed))
    width = np.full(model.tsr_class._DOF, half_width)
    return model.tsr_class(Bw=np.column_stack([centre - width, centre + width]))


def spanned_primitive(model, arm_count: int, q: np.ndarray):
    """The circle / sphere the end-effectors currently span, for drawing."""
    if arm_count not in (3, 4):
        return None
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    coordinates = zero.to_bw(model.forward_kinematics(q))
    return zero._primitive_from_bw(coordinates)


class NoCollision:
    """Free space -- only for comparing against a real checker."""

    def is_valid(self, q) -> bool:  # noqa: D102
        return True


def collision_checker(system, model, enabled: bool = True, radius: float = 0.07):
    """Capsule self-collision checker for the array, or a free-space stub.

    The arms face inward and their workspaces overlap on purpose, so without a
    checker the planner routes one arm straight through another: at radius 0.9,
    12% of configurations near the seed already interpenetrate.
    """
    if not enabled:
        return NoCollision()
    from pycbirrt.collision import CapsuleSelfCollisionChecker

    return CapsuleSelfCollisionChecker(system, model, radius=radius)


# Per-arm joint offsets that define the goal, cycled over the arm's joints.
# Deliberately not tiny: the bimanual solver does *worse* on small displacements
# from the tucked seed (0/15 region samples at 0.20, 14/15 at 0.35), because a
# near-identity target sits close to a singular posture.
_GOAL_OFFSETS = (0.45, -0.35, 0.30, -0.25, 0.20, -0.15)


def goal_configuration(model, arm_count: int, q_start: np.ndarray) -> np.ndarray:
    """A distinct, reachable goal configuration for this arm count.

    Every arm moves, so the array ends up holding the object somewhere visibly
    different. Defining the goal as a *configuration* (and the goal region as
    the pose it reaches) keeps it reachable by construction, which matters most
    for the coupled cases.

    The offsets are spread across each arm's joints rather than concentrated in
    the base. For a two-arm system, swinging both bases by the same amount turns
    out to be a pose the bimanual IK struggles to recover from a cold start,
    while a mixed displacement of the same size is solved reliably.
    """
    q_goal = np.asarray(q_start, dtype=float).copy()
    joints_per_arm = model.dof // max(arm_count, 1)
    for arm in range(arm_count):
        base = arm * joints_per_arm
        for joint in range(joints_per_arm):
            # Alternate the sign per arm so the arms do not all swing together.
            sign = 1.0 if arm % 2 == 0 else -1.0
            q_goal[base + joint] += sign * _GOAL_OFFSETS[joint % len(_GOAL_OFFSETS)]
    return q_goal


def goal_tsr(model, arm_count: int, q_goal: np.ndarray,
             position_tolerance: float = 0.04, angle_tolerance: float = 0.05):
    """A goal region of the right kind, centred on what ``q_goal`` reaches.

    The point of the example: the *shape* of the goal changes with the arm count
    even though the task ("get the array over there") does not.

      1 arm   a plain TSR on the end-effector Motor
      2 arms  a BimanualTSR: absolute pose free-ish, relative grasp held tight
      3 arms  a CircleTSR over [tx,ty,tz, dilation, n1,n2]
      4 arms  a SphereTSR over [tx,ty,tz, dilation]
    """
    pose = model.forward_kinematics(q_goal)

    if arm_count == 1:
        from tsr import TSR

        Bw = np.array([[-angle_tolerance, angle_tolerance]] * 3
                      + [[-position_tolerance, position_tolerance]] * 3)
        return TSR(T0_w=pose.to_transformation_matrix(), Tw_e=np.eye(4), Bw=Bw)

    if arm_count == 2:
        from tsr import TSR
        from tsr.bimanual import BimanualTSR

        # Absolute: where the pair holds the object. Relative: the grasp itself,
        # pinned tight so the two hands keep their relative transform.
        absolute = TSR(T0_w=pose.absolute.to_transformation_matrix(), Tw_e=np.eye(4),
                       Bw=np.array([[-angle_tolerance, angle_tolerance]] * 3
                                   + [[-position_tolerance, position_tolerance]] * 3))
        relative = TSR(T0_w=pose.relative.to_transformation_matrix(), Tw_e=np.eye(4),
                       Bw=np.array([[-0.02, 0.02]] * 3 + [[-0.01, 0.01]] * 3))
        return BimanualTSR(absolute=absolute, relative=relative)

    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    centre = zero.to_bw(pose)
    width = np.full(model.tsr_class._DOF, position_tolerance)
    width[3] = 0.03                        # dilation: hold the shape's size
    if model.tsr_class._DOF > 4:
        width[4:] = angle_tolerance        # the circle's plane normal
    return model.tsr_class(Bw=np.column_stack([centre - width, centre + width]))


def joint_travel(path) -> float:
    """Total joint motion along a path, in radians.

    Differences are wrapped to (-pi, pi]: these are revolute joints, so a step
    that crosses the +/-pi seam is a small motion, not a full turn. Summing raw
    differences reports a 2*pi jump for what the robot does as a 0.05 rad step.
    """
    steps = np.diff(np.asarray(path, dtype=float), axis=0)
    return float(np.abs(np.arctan2(np.sin(steps), np.cos(steps))).sum())


def ik_solver_for(model, arm_count: int, collision):
    """The IK solver matching this arm count."""
    if arm_count == 1:
        from pycbirrt.backends.gafro import GafroIKSolver

        return GafroIKSolver(model.manipulator, model.joint_limits, max_iterations=200,
                             tolerance=1e-5, base_configuration=model.base_configuration,
                             collision_checker=collision)
    if arm_count == 2:
        from pycbirrt.backends.gafro_bimanual import GafroBimanualIKSolver

        return GafroBimanualIKSolver(model, collision_checker=collision,
                                     max_iterations=200, tolerance=1e-4)
    from pycbirrt.backends.gafro_multiarm import GafroMultiArmIKSolver

    return GafroMultiArmIKSolver(model, max_iterations=150, tolerance=1e-4,
                                 collision_checker=collision)


def plan_to_goal(model, arm_count: int, q_start: np.ndarray, metric=None,
                 metric_sampling: bool = False, geodesic_extension: bool = False,
                 seed: int = 1, attempts: int = 1, collision=None):
    """Plan start -> goal region for any arm count.

    ``attempts`` re-seeds and retries: the coupled models (two arms especially)
    depend on IK reaching a configuration that satisfies every pose component at
    once, and that can miss from an unlucky tree seed.

    Returns ``(path, goal_region, q_goal)``; ``path`` is None if none was found.
    """
    from pycbirrt import CBiRRT, CBiRRTConfig

    collision = collision if collision is not None else NoCollision()
    q_goal = goal_configuration(model, arm_count, q_start)
    region = goal_tsr(model, arm_count, q_goal)
    solver = ik_solver_for(model, arm_count, collision)
    config = CBiRRTConfig(max_iterations=2000, step_size=0.2, goal_bias=0.3,
                          tsr_samples=30, angular_joints=(True,) * model.dof,
                          metric=metric, metric_sampling=metric_sampling,
                          geodesic_extension=geodesic_extension)
    from pycbirrt.exceptions import PlanningError

    planner = CBiRRT(model, solver, collision, config)
    for attempt in range(max(attempts, 1)):
        try:
            result = planner.plan(start=q_start, goal_tsrs=[region],
                                  seed=seed + attempt, return_details=True)
        except PlanningError:
            continue
        if result.success:
            return result.path, region, q_goal
    return None, region, q_goal


def plan_bigger_shape(model, arm_count: int, q_seed: np.ndarray, growth: float = 0.25,
                      metric=None, metric_sampling: bool = False,
                      geodesic_extension: bool = False, collision=None):
    """Plan a path that grows the held circle / sphere by ``growth`` in log scale.

    ``metric`` is the configuration-space metric the planner measures with; None
    means the Euclidean default. A kinetic-energy metric matters more here than
    for one arm: the array has several arms' worth of inertia, and swinging a
    whole arm inward costs far more than flicking one wrist.

    Returns ``(path, region)``, or ``(None, region)`` if no path was found.
    """
    from pycbirrt import CBiRRT, CBiRRTConfig
    from pycbirrt.backends.gafro_multiarm import GafroMultiArmIKSolver

    collision = collision if collision is not None else NoCollision()
    dof = model.tsr_class._DOF
    zero = model.tsr_class(Bw=np.zeros((dof, 2)))
    base = zero.to_bw(model.forward_kinematics(q_seed))
    goal = base.copy()
    goal[3] += growth
    width = np.full(dof, 0.03)
    width[3] = 0.02
    region = model.tsr_class(Bw=np.column_stack([goal - width, goal + width]))

    solver = GafroMultiArmIKSolver(model, max_iterations=150, tolerance=1e-4,
                                   collision_checker=collision)
    config = CBiRRTConfig(max_iterations=800, step_size=0.25, goal_bias=0.3,
                          tsr_samples=15, angular_joints=(True,) * model.dof,
                          metric=metric, metric_sampling=metric_sampling,
                          geodesic_extension=geodesic_extension)
    planner = CBiRRT(model, solver, collision, config)
    result = planner.plan(start=q_seed, goal_tsrs=[region], seed=1, return_details=True)
    return (result.path if result.success else None), region


def visualize(model, system, arm_count: int, q_seed: np.ndarray, label: str, port: int,
              path=None):
    """Serve a viser scene: the arm array, the spanned primitive, a dilation slider."""
    _patch_visualizer_joint_limits()
    viz = ga.Visualizer(port=port)
    robot_viz = viz.add_robot(system, joint_sliders=False)
    robot_viz.update(to_system(model, q_seed))

    state = {"q": q_seed.copy(), "node": None}

    def redraw():
        primitive = spanned_primitive(model, arm_count, state["q"])
        if primitive is None:
            return
        if state["node"] is not None:
            state["node"].remove()
        add = viz.add_sphere if arm_count == 4 else viz.add_circle
        state["node"] = add(primitive, name="/spanned", color=PRIMITIVE_COLOR, opacity=0.3)

    redraw()
    viz.add_label(f"{arm_count} arm(s) - {label}", position=(0.0, 0.0, 0.9), name="/title")

    if path:
        def on_frame(_index, q):
            state["q"] = np.asarray(q, dtype=float)
            robot_viz.update(to_system(model, state["q"]))
            redraw()

        viz.add_playback(list(path), on_frame)

    if arm_count in (3, 4):
        from pycbirrt.backends.gafro_multiarm import GafroMultiArmIKSolver

        solver = GafroMultiArmIKSolver(model, max_iterations=400, tolerance=1e-5)
        zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
        base = zero.to_bw(model.forward_kinematics(q_seed))

        with viz.gui.add_folder("Held shape"):
            slider = viz.gui.add_slider("dilation (log scale)", min=-0.6, max=0.6,
                                        step=0.02, initial_value=0.0)
            status = viz.gui.add_text("status", initial_value="-", disabled=True)

        def on_dilation(_event=None):
            goal = base.copy()
            goal[3] = base[3] + slider.value
            width = np.array([0.02] * 3 + [0.005]
                             + [0.05] * (model.tsr_class._DOF - 4))
            region = model.tsr_class(Bw=np.column_stack([goal - width, goal + width]))
            solutions = solver.solve(region, q_init=state["q"])
            if not solutions:
                status.value = f"no IK at dilation {slider.value:+.2f}"
                return
            state["q"] = solutions[0]
            robot_viz.update(to_system(model, state["q"]))
            redraw()
            achieved = zero.to_bw(model.forward_kinematics(state["q"]))[3]
            status.value = f"scale x{np.exp(achieved - base[3]):.3f}"

        slider.on_update(on_dilation)

    viz.show()


def to_system(model, q: np.ndarray) -> np.ndarray:
    """Controlled-width q -> System-width config for the visualizer.

    Models differ in how much of the System they cover: a single-arm model
    scatters its chain into the full config, the multi-arm models hold their
    non-controlled joints, and the bimanual model already plans in System width.
    """
    if hasattr(model, "to_system_configuration"):
        return model.to_system_configuration(q)
    if hasattr(model, "_to_task_full"):
        return model._to_task_full(q)
    return np.asarray(q, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", type=int, default=4, choices=(1, 2, 3, 4),
                        help="number of manipulators on the circle (default: 4)")
    parser.add_argument("--radius", type=float, default=0.7,
                        help="circle radius the arms are mounted on (default: 0.7)")
    parser.add_argument("--port", type=int, default=8080, help="viser port (default: 8080)")
    parser.add_argument("--no-viz", action="store_true", help="print only, no viewer")
    parser.add_argument("--out-dir", default=None,
                        help="where to write the composed array (default: a temp dir)")
    parser.add_argument("--plan", action="store_true",
                        help="plan a path from the seed configuration to a goal region")
    parser.add_argument("--no-self-collision", action="store_true",
                        help="disable capsule self-collision checking between arms")
    parser.add_argument("--capsule-radius", type=float, default=0.07,
                        help="capsule radius used for self-collision (default: 0.07 m)")
    parser.add_argument("--attempts", type=int, default=6,
                        help="re-seeded planning attempts (two arms needs a few)")
    parser.add_argument("--task", default="move", choices=("move", "grow"),
                        help="move: plan to a distinct goal pose (all arm counts); "
                             "grow: enlarge the held shape (3 and 4 arms only)")
    parser.add_argument("--metric", default="euclidean", choices=("euclidean", "kinetic"),
                        help="configuration-space metric the planner measures with")
    parser.add_argument("--metric-sampling", action="store_true",
                        help="weight random samples by sqrt(det M) (needs --metric kinetic)")
    parser.add_argument("--geodesic-extension", action="store_true",
                        help="extend along the natural gradient (needs --metric kinetic; "
                             "known not to connect for --arms 2)")
    args = parser.parse_args()

    system, path = build_system(args.arms, args.radius, args.out_dir)
    model, label = build_model(args.arms, system)
    q_seed = seed_configuration(model, args.arms)

    print(f"array:      {args.arms} x UR5e on a circle of radius {args.radius} m")
    print(f"written to: {path}")
    print(f"system dof: {system.get_dof()}   planning dof: {model.dof}")
    print(f"task space: {label}")
    print(f"pose:       {describe_pose(model, args.arms, q_seed)}")

    region = plan_region(model, args.arms, q_seed)
    if region is not None:
        pose = model.forward_kinematics(q_seed)
        print(f"region:     {region}")
        print(f"            contains seed pose: {region.contains(pose, tolerance=1e-6)}")

    path = None
    if args.plan:
        metric = None
        if args.metric == "kinetic":
            from pycbirrt.metrics import metric_for_model

            metric = metric_for_model(model)

        collision = collision_checker(system, model,
                                      enabled=not args.no_self_collision,
                                      radius=args.capsule_radius)
        if hasattr(collision, "clearance"):
            print(f"collision:  capsule self-check on, "
                  f"seed clearance {collision.clearance(q_seed):+.3f} m")
        else:
            print("collision:  OFF (arms may pass through each other)")

        if args.task == "grow":
            if args.arms < 3:
                print("plan:       --task grow needs a held shape (3 or 4 arms); "
                      "use --task move")
            else:
                path, _region = plan_bigger_shape(
                    model, args.arms, q_seed, metric=metric,
                    metric_sampling=args.metric_sampling,
                    geodesic_extension=args.geodesic_extension,
                    collision=collision)
                if path is None:
                    print("plan:       no path found")
                else:
                    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
                    start_dilation = zero.to_bw(model.forward_kinematics(q_seed))[3]
                    end_dilation = zero.to_bw(model.forward_kinematics(path[-1]))[3]
                    print(f"plan:       [{args.metric}] {len(path)} waypoints, "
                          f"dilation {start_dilation:+.3f} -> {end_dilation:+.3f} "
                          f"(x{np.exp(end_dilation - start_dilation):.2f} bigger)")
        else:
            path, region, q_goal = plan_to_goal(
                model, args.arms, q_seed, metric=metric,
                metric_sampling=args.metric_sampling,
                geodesic_extension=args.geodesic_extension,
                attempts=args.attempts, collision=collision)
            print(f"goal:       {region}")
            if path is None:
                print("plan:       no path found")
            else:
                reached = model.forward_kinematics(path[-1])
                travel = joint_travel(path)
                print(f"plan:       [{args.metric}] {len(path)} waypoints, "
                      f"joint travel {travel:.3f} rad")
                print(f"            start: {describe_pose(model, args.arms, q_seed)}")
                print(f"            end:   {describe_pose(model, args.arms, path[-1])}")
                # distance() is the one predicate every region kind shares
                # (TSR.contains takes no tolerance; the others do).
                distance, _witness = region.distance(reached)
                print(f"            distance to goal region: {distance:.2e}")
                if hasattr(collision, "clearance"):
                    worst = min(collision.clearance(w) for w in path)
                    print(f"            min arm-to-arm clearance: {worst:+.3f} m")

    if args.no_viz:
        return 0

    visualize(model, system, args.arms, q_seed, label, args.port, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
