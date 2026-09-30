# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""pick: the goal is a set, every side grasp of any of three cans, and one call plans to it."""

from __future__ import annotations

import mujoco

from sscbirrt import CBiRRTConfig
from sscbirrt.demo.grasps import side_grasps
from sscbirrt.demo.render import Camera, Clip
from sscbirrt.demo.scenarios import Outcome, Scenario
from sscbirrt.demo.scene import EE_SITE, HOME, UR5E_JOINTS, build_scene, can_position, set_arm, ur5e_xml
from sscbirrt.mujoco import Arm, plan

CANS = {
    "red can": can_position((0.45, 0.22)),
    "blue can": can_position((0.62, -0.05)),
    "yellow can": can_position((0.40, -0.25)),
}


def run(seed: int) -> Outcome:
    model = build_scene({name.replace(" ", "_"): pos for name, pos in CANS.items()})
    data = mujoco.MjData(model)
    set_arm(model, data, HOME)
    arm = Arm(model, UR5E_JOINTS, EE_SITE, mjcf=ur5e_xml())

    goals, owner = [], []
    for name, center in CANS.items():
        regions = side_grasps(center)
        goals += regions
        owner += [name] * len(regions)

    config = CBiRRTConfig(step_size=0.2, edge_resolution=0.05, num_tree_roots=20, timeout=30.0)
    result = plan(model, data, arm, goal=goals, config=config, seed=seed)

    report = [f"goal set: {len(goals)} side-grasp regions of 3 cans (tsr.Robotiq2F85 cylinder primitive)"]
    if not result.success:
        return Outcome(False, report + [f"no path: {result.failure_reason}"], model, data)
    reached = owner[result.goal_index]
    report += [
        f"reached: the {reached} (goal_index {result.goal_index})",
        f"backend: {result.backend}, {result.planning_time:.2f} s, {len(result.path)} waypoints, seed {seed}",
    ]
    report += [f"  not native because: {r}" for r in result.backend_reasons]
    if result.backend == "native":
        prov = result.provenance
        report.append(
            f"provenance: SSIK {prov.get('ssik_solver_name')}, snapshot {prov.get('snapshot_sha256', '')[:12]}"
        )

    clip = Clip(
        path=result.path,
        title="One goal set, many ways to reach it",
        lines=[
            f"Goal: any side grasp of any can ({len(goals)} regions)",
            f"Planner chose: the {reached}",
            f"{result.backend} backend, planned in {result.planning_time:.2f} s",
        ],
    )
    return Outcome(True, report, model, data, [clip], Camera())


SCENARIO = Scenario(
    name="pick",
    claim="a goal can be a set: one call plans to whichever side grasp of whichever can is reachable",
    run=run,
)
