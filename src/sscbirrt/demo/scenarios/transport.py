# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""transport: carry a held can across the table twice, free and then kept upright by a path constraint."""

from __future__ import annotations

import mujoco
import numpy as np
from tsr import TSR, Robotiq2F85

from sscbirrt import CBiRRTConfig
from sscbirrt.demo.render import Camera, Clip
from sscbirrt.demo.scenarios import Outcome, Scenario
from sscbirrt.demo.scene import (
    CAN_HALF_HEIGHT,
    CAN_RADIUS,
    EE_SITE,
    GRIPPER_BODY,
    HOME,
    TABLE_TOP_Z,
    UR5E_JOINTS,
    build_scene,
    can_position,
    set_arm,
    ur5e_xml,
)
from sscbirrt.mujoco import Arm, plan

CAN = "green_can"
PICK_AT = can_position((0.45, 0.25))
PLACE_AT = can_position((0.50, -0.25))
OBSTACLES = {"box_on_table": ((0.52, 0.0, TABLE_TOP_Z + 0.09), (0.06, 0.06, 0.09))}
TILT_LIMIT = 0.05  # rad
RUNS = 10  # seeds per carry; the report gives the spread, the video the worst free carry


def _grasp(center: np.ndarray) -> TSR:
    """One side grasp of a can at ``center``: mid depth, fingers level (roll 0), at mid height, any approach angle.

    The same grasp at pick and place, because the held can keeps its pose in the gripper. The height is pinned so
    that every member holds the can the same way: the can is symmetric about its axis, so the approach angle does
    not change the grasp, and ``HELD`` is exact for all of them.
    """
    T_bottom = np.eye(4)
    T_bottom[:3, 3] = np.asarray(center) - [0.0, 0.0, CAN_HALF_HEIGHT]
    templates = Robotiq2F85().grasp_cylinder_side(cylinder_radius=CAN_RADIUS, cylinder_height=2 * CAN_HALF_HEIGHT)
    mid_level = next(t for t in templates if "mid" in t.name and "roll 0" in t.name)
    region = mid_level.instantiate(T_bottom)
    bounds = np.array(region.Bw, dtype=float)
    bounds[2] = 0.0  # mid height only
    return TSR(T0_w=region.T0_w, Tw_e=region.Tw_e, Bw=bounds)


def _held(model: mujoco.MjModel) -> np.ndarray:
    """The can's pose in the gripper body's frame for this grasp, from the grasp geometry alone."""
    from sscbirrt.backends.mujoco import site_offset_in_body

    region = _grasp(np.zeros(3))
    T_grasp_can = np.linalg.inv(region.T0_w @ region.Tw_e)  # the can's center is the region's origin here
    return site_offset_in_body(model, EE_SITE) @ T_grasp_can


def upright() -> TSR:
    """Held can upright: the grasp frame's x axis (the can's axis) within TILT_LIMIT of vertical, any heading,
    above the table."""
    Tw_e = np.eye(4)
    Tw_e[:3, :3] = [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]  # grasp x -> world z, as in a roll-0 grasp
    bounds = [
        [-2, 2],
        [-2, 2],
        [TABLE_TOP_Z, 1.5],
        [-TILT_LIMIT, TILT_LIMIT],
        [-TILT_LIMIT, TILT_LIMIT],
        [-np.pi, np.pi],
    ]
    return TSR(T0_w=np.eye(4), Tw_e=Tw_e, Bw=bounds)


def tilt(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> float:
    """How far the held can leans from vertical at ``q``, in radians."""
    set_arm(model, data, q)
    axis = data.site(EE_SITE).xmat.reshape(3, 3)[:, 0]
    return float(np.arccos(np.clip(axis[2], -1.0, 1.0)))


def run(seed: int) -> Outcome:
    model = build_scene({CAN: PICK_AT}, OBSTACLES)
    data = mujoco.MjData(model)
    set_arm(model, data, HOME)
    arm = Arm(model, UR5E_JOINTS, EE_SITE, mjcf=ur5e_xml())
    config = CBiRRTConfig(step_size=0.2, edge_resolution=0.05, num_tree_roots=20, timeout=30.0)

    held = {CAN: (GRIPPER_BODY, _held(model))}
    view = mujoco.MjData(model)

    def max_tilt(path) -> float:
        return max(tilt(model, view, q) for q in path)

    # Upright first, starting from the set of all grasps of the can: the planner picks a grasp configuration from
    # which an upright carry exists (from some windings of the wrist there is none within the joint limits).
    uprights = []
    for k in range(RUNS):
        r = plan(
            model, data, arm, start=_grasp(PICK_AT), goal=_grasp(PLACE_AT), constraint=upright(), holding=held,
            config=config, seed=seed + k,
        )  # fmt: skip
        if not r.success:
            return Outcome(False, [f"upright (seed {seed + k}): no path: {r.failure_reason}"], model, data)
        uprights.append(r)
    carried = uprights[0]
    q_grasp = carried.path[0]

    # Then free carries from the same grasp. Whether one free plan tilts the can depends on its seed, so plan
    # several and show the worst: the point is that nothing keeps the can upright, not that every plan spills it.
    frees = []
    for k in range(RUNS):
        r = plan(model, data, arm, start=q_grasp, goal=_grasp(PLACE_AT), holding=held, config=config, seed=seed + k)
        if not r.success:
            return Outcome(False, [f"free (seed {seed + k}): no path: {r.failure_reason}"], model, data)
        frees.append((max_tilt(r.path), r))
    free_tilts = [t for t, _ in frees]
    worst_free_tilt, free = max(frees, key=lambda pair: pair[0])
    upright_tilts = [max_tilt(r.path) for r in uprights]
    spilled = sum(t > np.radians(30) for t in free_tilts)

    report = [
        f"free carries:    {spilled} of {RUNS} tilt the can past 30 deg; "
        f"worst {np.degrees(worst_free_tilt):.0f} deg (shown)",
        f"upright carries: {RUNS} of {RUNS} within the constraint; worst {np.degrees(max(upright_tilts)):.1f} deg",
        f"backend: {carried.backend}; upright plans {min(r.planning_time for r in uprights):.2f}-"
        f"{max(r.planning_time for r in uprights):.2f} s, free {max(r.planning_time for _, r in frees):.2f} s at most",
        "start: chosen from every grasp of the can, so that an upright carry exists",
    ]

    def clip(carry, title, lines):
        return Clip(
            path=carry.path,
            title=title,
            lines=lines,
            caption=lambda q: f"tilt {np.degrees(tilt(model, view, q)):5.1f} deg",
            held=(CAN, held[CAN][1]),
        )

    clips = [
        clip(
            free,
            "Carried freely",
            [
                f"No constraint: {spilled} of {RUNS} plans tilt the can past 30 deg",
                f"The worst of them, shown: up to {np.degrees(worst_free_tilt):.0f} deg",
            ],
        ),
        clip(
            carried,
            "Carried upright: a path constraint",
            [
                f"Constraint: the can's roll and pitch within {np.degrees(TILT_LIMIT):.0f} deg",
                f"{RUNS} of {RUNS} plans hold it; worst tilt {np.degrees(max(upright_tilts)):.1f} deg",
                f"{carried.backend} backend, planned in {carried.planning_time:.2f} s",
            ],
        ),
    ]
    set_arm(model, data, q_grasp)
    return Outcome(True, report, model, data, clips, Camera())


SCENARIO = Scenario(
    name="transport",
    claim="a path constraint holds everywhere: the same carry, free and then with the can kept upright",
    run=run,
)
