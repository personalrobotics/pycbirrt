# SPDX-License-Identifier: MIT
"""Build a System with N copies of one manipulator arranged in a circle.

gafro's YAML robot description is a flat structural document (links, joints,
actuators, kinematic chains). Instancing an arm N times is therefore a matter of
namespacing every name and giving each copy its own base pose on a circle:

    arm0 at angle 0, arm1 at 2*pi/N, ...

all facing inward toward the circle's centre, so their workspaces overlap and a
cooperative task space over them is meaningful.

The result is written as a YAML file that ``SystemSerialization.load`` reads,
which keeps this independent of the C++ System-building API.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import yaml

# One arm of the array; any single-chain description with this structure works.
# Set $CIRCLE_ARRAY_SOURCE to a different description to run this elsewhere.
# (Deliberately not $GAFRO_ROBOT_DESCRIPTIONS: that already means the *assets*
# directory in some environments, so reusing it would double the path.)
_DEFAULT_ASSETS = Path(
    "/home/tobi/tmp/vvv/gafro-robot-descriptions/assets/robots/universal_robots/ur5e/ur5e.yaml")
DEFAULT_SOURCE = Path(os.environ.get("CIRCLE_ARRAY_SOURCE", _DEFAULT_ASSETS))
# The chain whose tip is the end-effector (per arm, before namespacing).
TIP_CHAIN = "ur5e_ee"
# Task space spanning every arm's tip chain; GafroBimanualModel looks its
# cooperative task space up on the System by name.
COOPERATIVE_TASK_SPACE = "coop"
# Massless link every arm is mounted on; the root of the merged tree.
WORLD_LINK = "world"


def _pose(x: float, y: float, z: float) -> dict:
    return {"position": {"x": x, "y": y, "z": z},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


def _quaternion_about_z(angle: float) -> dict:
    return {"w": math.cos(angle / 2.0), "x": 0.0, "y": 0.0, "z": math.sin(angle / 2.0)}


def _rename(value, prefix: str) -> str:
    return f"{prefix}/{value}"


def _namespace_arm(system: dict, prefix: str) -> dict:
    """One arm instance with every name prefixed."""
    arm = copy.deepcopy(system)
    arm["name"] = prefix
    arm["root_link"] = _rename(arm["root_link"], prefix)

    arm["links"] = {
        _rename(name, prefix): _relabel_link(link, prefix)
        for name, link in arm["links"].items()
    }
    arm["joints"] = {
        _rename(name, prefix): _relabel_joint(joint, prefix)
        for name, joint in arm.get("joints", {}).items()
    }
    arm["actuators"] = [_relabel_actuator(a, prefix) for a in arm.get("actuators", [])]
    arm["kinematic_chains"] = [
        {"name": _rename(c["name"], prefix), "tip": _rename(c["tip"], prefix)}
        for c in arm.get("kinematic_chains", [])
    ]
    arm["task_spaces"] = [
        {"name": _rename(t["name"], prefix),
         "chains": [_rename(c, prefix) for c in t.get("chains", [])]}
        for t in arm.get("task_spaces", [])
    ]
    return arm


def _relabel_link(link: dict, prefix: str) -> dict:
    link = copy.deepcopy(link)
    link["name"] = _rename(link["name"], prefix)
    if "parent_joint" in link:
        link["parent_joint"] = _rename(link["parent_joint"], prefix)
    if "joints" in link:
        link["joints"] = [_rename(j, prefix) for j in link["joints"]]
    return link


def _relabel_joint(joint: dict, prefix: str) -> dict:
    joint = copy.deepcopy(joint)
    joint["name"] = _rename(joint["name"], prefix)
    for key in ("parent_link", "child_link"):
        if key in joint:
            joint[key] = _rename(joint[key], prefix)
    return joint


def _relabel_actuator(actuator: dict, prefix: str) -> dict:
    actuator = copy.deepcopy(actuator)
    actuator["name"] = _rename(actuator["name"], prefix)
    if "joint" in actuator:
        actuator["joint"] = _rename(actuator["joint"], prefix)
    return actuator


def build_circle_array(arm_count: int, radius: float = 0.9,
                       source: Path = DEFAULT_SOURCE,
                       name: str | None = None) -> dict:
    """Compose ``arm_count`` copies of ``source`` evenly spaced on a circle.

    Each arm sits at radius ``radius`` from the origin and is yawed to face
    inward, so all end-effectors can meet near the centre.
    """
    if arm_count < 1:
        raise ValueError(f"arm_count must be >= 1, got {arm_count}")
    source = Path(source)
    single = yaml.safe_load(source.read_text())["system"]
    # meshdir is relative to the *source* description; the composed array is
    # written elsewhere, so resolve it to an absolute path or the visualizer
    # looks for the meshes next to the output file and fails to load them.
    mesh_dir = Path(single.get("meshdir", "assets"))
    if not mesh_dir.is_absolute():
        mesh_dir = (source.parent / mesh_dir).resolve()

    merged = {
        "name": name or f"circle_array_{arm_count}",
        "meshdir": str(mesh_dir),
        # A single massless world link is the tree root; each arm's own base is
        # bolted to it by a fixed joint carrying that arm's place on the circle.
        # Without this the arms are disconnected components and the loader
        # cannot resolve any chain tip beyond the first.
        "root_link": WORLD_LINK,
        "base_pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0},
                      "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}},
        "kinematic_chains": [],
        "task_spaces": [],
        "links": {},
        "joints": {},
        "actuators": [],
    }
    if "attributes" in single:
        merged["attributes"] = copy.deepcopy(single["attributes"])
        # The source's keyframes are sized for one arm and no longer apply.
        merged["attributes"].pop("keyframes", None)

    merged["links"][WORLD_LINK] = {
        "name": WORLD_LINK,
        "inertial": {
            "mass": 0.0,
            "inertia": {"xx": 0.0, "xy": 0.0, "xz": 0.0, "yy": 0.0, "yz": 0.0, "zz": 0.0},
            "origin": _pose(0.0, 0.0, 0.0),
        },
        "joints": [],
    }

    for index in range(arm_count):
        angle = 2.0 * math.pi * index / arm_count
        prefix = f"arm{index}"
        arm = _namespace_arm(single, prefix)

        # Fixed joint world -> this arm's base, carrying its pose on the circle.
        # Face inward: yaw by angle + pi so the arm's +x points at the centre.
        mount = f"{prefix}/mount"
        merged["joints"][mount] = {
            "type": "fixed",
            "name": mount,
            "parent_link": WORLD_LINK,
            "child_link": arm["root_link"],
            "origin": {
                "position": {"x": radius * math.cos(angle),
                             "y": radius * math.sin(angle),
                             "z": 0.0},
                "orientation": _quaternion_about_z(angle + math.pi),
            },
        }
        merged["links"][WORLD_LINK]["joints"].append(mount)
        arm["links"][arm["root_link"]]["parent_joint"] = mount

        merged["links"].update(arm["links"])
        merged["joints"].update(arm["joints"])
        merged["actuators"].extend(arm["actuators"])
        merged["kinematic_chains"].extend(arm["kinematic_chains"])
        merged["task_spaces"].extend(arm["task_spaces"])

    # A cooperative task space over every arm, for the models that take one by
    # name rather than building it from a chain list.
    merged["task_spaces"].append({
        "name": COOPERATIVE_TASK_SPACE,
        "chains": chain_names(arm_count, tip_chain=TIP_CHAIN),
    })

    return {"system": merged}


def chain_names(arm_count: int, tip_chain: str = TIP_CHAIN) -> list[str]:
    """End-effector chain name of each arm, in array order."""
    return [f"arm{i}/{tip_chain}" for i in range(arm_count)]


def write_circle_array(arm_count: int, destination: Path, radius: float = 0.9,
                       source: Path = DEFAULT_SOURCE) -> Path:
    """Write the composed array to ``destination`` and return the path."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(build_circle_array(arm_count, radius, source),
                                          sort_keys=False))
    return destination
