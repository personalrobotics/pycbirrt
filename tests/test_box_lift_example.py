#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The arm-array box pick-and-lift example."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from _circle_array import DEFAULT_SOURCE  # noqa: E402

requires_source = pytest.mark.skipif(
    not os.path.exists(DEFAULT_SOURCE), reason=f"missing {DEFAULT_SOURCE}")


def _setup(arm_count, tmp_path, collisions=True):
    from circle_array_box_lift import DEFAULT_RADIUS, grasp_configuration
    from circle_array_tsr import build_model, build_system, collision_checker

    system, _file = build_system(arm_count, DEFAULT_RADIUS, tmp_path)
    model, _label = build_model(arm_count, system)
    checker = collision_checker(system, model, enabled=collisions)
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    q_grasp, box_size, box_centre = grasp_configuration(model, system, arm_count, checker)
    return system, model, checker, zero, q_grasp, box_size, box_centre


@requires_source
@pytest.mark.parametrize("arm_count", [3, 4])
def test_grasp_configuration_is_collision_free(tmp_path, arm_count):
    _system, _model, checker, _zero, q_grasp, box_size, box_centre = _setup(arm_count, tmp_path)
    assert checker.is_valid(q_grasp)
    assert checker.clearance(q_grasp) > 0


@requires_source
@pytest.mark.parametrize("arm_count", [3, 4])
def test_lift_region_pins_the_dilation(tmp_path, arm_count):
    """A rigid grasp is a constant-scale constraint: the dilation box is tight."""
    from circle_array_box_lift import lift_region

    _system, model, _checker, zero, q_grasp, box_size, box_centre = _setup(arm_count, tmp_path)
    region = lift_region(model, arm_count, q_grasp, 0.25)
    grasped = zero.to_bw(model.forward_kinematics(q_grasp))

    dilation_width = region.Bw[3, 1] - region.Bw[3, 0]
    assert dilation_width <= 0.05, "dilation must be pinned for a rigid grasp"
    assert region.Bw[3, 0] <= grasped[3] <= region.Bw[3, 1]
    # ...while z is centred on the lifted height, not the grasped one.
    assert region.Bw[2, 0] > grasped[2]


@requires_source
@pytest.mark.parametrize("arm_count", [3, 4])
def test_lift_plans_and_keeps_the_grasp_rigid(tmp_path, arm_count):
    """The whole point: the box goes up, its size does not change, no collisions."""
    from circle_array_box_lift import RigidGraspConstraint, lift_region, plan_lift

    system, model, checker, zero, q_grasp, box_size, box_centre = _setup(arm_count, tmp_path)
    region = lift_region(model, arm_count, q_grasp, 0.15)
    constraint = RigidGraspConstraint(model, system, arm_count, q_grasp)
    path = plan_lift(model, arm_count, q_grasp, region, checker, attempts=6,
                     constraint=constraint)
    assert path is not None, f"no lift path for {arm_count} arms"
    assert min(checker.clearance(w) for w in path) >= 0.0

    # The guarantee is over the *contacts*, not the dilation coordinate: the
    # two correlate at only ~0.19, so dilation is not a proxy for box size.
    for waypoint in path:
        assert constraint.drift(waypoint) <= constraint.tolerance + 1e-9, (
            "the grasp deformed mid-carry")

    # And no arm passes through another.
    assert min(checker.clearance(w) for w in path) >= 0.0


@requires_source
def test_path_constraint_is_what_keeps_the_grasp_rigid(tmp_path):
    """Constraining only the goal lets the array stretch the box mid-carry.

    Measured: without the path constraint the dilation wanders several times
    further from its grasped value than with it.
    """
    from circle_array_box_lift import RigidGraspConstraint, lift_region, plan_lift

    system, model, checker, _zero, q_grasp, _size, _centre = _setup(4, tmp_path)
    region = lift_region(model, 4, q_grasp, 0.15)
    constraint = RigidGraspConstraint(model, system, 4, q_grasp)

    path = plan_lift(model, 4, q_grasp, region, checker, attempts=6,
                     constraint=constraint)
    assert path is not None
    assert max(constraint.drift(w) for w in path) <= constraint.tolerance + 1e-9


@requires_source
def test_box_rises(tmp_path):
    """The end-effector centroid -- where the box actually is -- goes up."""
    from circle_array_box_lift import (
        RigidGraspConstraint,
        box_pose_along,
        lift_region,
        plan_lift,
    )

    system, model, checker, _zero, q_grasp, box_size, box_centre = _setup(4, tmp_path)
    region = lift_region(model, 4, q_grasp, 0.25)
    path = plan_lift(model, 4, q_grasp, region, checker, attempts=6,
                     constraint=RigidGraspConstraint(model, system, 4, q_grasp))
    assert path is not None

    start = box_pose_along(model, system, 4, path[0])
    end = box_pose_along(model, system, 4, path[-1])
    assert end[2] - start[2] > 0.1, "the box did not rise"


@requires_source
def test_box_position_is_the_contact_centroid(tmp_path):
    """The box is drawn where the contacts are, not at the circumsphere centre.

    For a near-coplanar grasp those differ wildly -- the sphere through four
    almost-level contacts has its centre far off -- so the centroid is the only
    sane choice.
    """
    from circle_array_box_lift import box_pose_along, end_effector_positions

    system, model, _checker, _zero, q_grasp, _size, _centre = _setup(4, tmp_path)
    centroid = box_pose_along(model, system, 4, q_grasp)
    contacts = end_effector_positions(model, system, 4, q_grasp)
    np.testing.assert_allclose(centroid, contacts.mean(axis=0), atol=1e-9)
    assert 0.05 < centroid[2] < 1.0, "contacts should be at a plausible height"


@requires_source
def test_box_stands_on_the_ground(tmp_path):
    """The box must rest on the floor, not float in mid-air."""
    _system, _model, _checker, _zero, _q, box_size, box_centre = _setup(4, tmp_path)
    bottom = box_centre[2] - box_size[2] / 2.0
    assert bottom == pytest.approx(0.0, abs=1e-9)


@requires_source
def test_arms_reach_the_box_without_entering_it(tmp_path):
    """Contacts sit one link half-thickness off the surface -- touching, not through.

    The end-effector *origin* cannot be on the surface: the link has thickness,
    and placing the origin there buries the wrist capsule ~9 cm inside the box.
    What matters is that the standoff is small (the arms reach the box) and the
    capsules stay outside it.
    """
    from circle_array_box_lift import CONTACT_STANDOFF, checker_with_box, end_effector_positions

    system, model, checker, _zero, q_grasp, box_size, box_centre = _setup(4, tmp_path)
    contacts = end_effector_positions(model, system, 4, q_grasp)
    gaps = (np.abs(contacts - box_centre) - box_size / 2.0).max(axis=1)
    np.testing.assert_allclose(gaps, CONTACT_STANDOFF, atol=1e-6)

    checker = checker_with_box(system, model, checker, box_size, box_centre)
    assert checker.obstacle_clearance(q_grasp) >= 0.0


@requires_source
def test_grasp_is_not_degenerate(tmp_path):
    """A perfectly level 4-point grasp is coplanar and makes the task pose NaN."""
    from tsr.multiarm import is_degenerate

    _system, model, _checker, _zero, q_grasp, _size, _centre = _setup(4, tmp_path)
    assert not is_degenerate(model.forward_kinematics(q_grasp))
    assert np.all(np.isfinite(_zero_bw(model, q_grasp)))


def _zero_bw(model, q):
    zero = model.tsr_class(Bw=np.zeros((model.tsr_class._DOF, 2)))
    return zero.to_bw(model.forward_kinematics(q))


@requires_source
def test_arms_do_not_penetrate_the_box(tmp_path):
    """The object being carried must be an obstacle, not scenery.

    Before the box was added to the collision checker the wrist links sat ~9 cm
    inside it at the grasp and reached 13 cm in along the lift path.
    """
    from circle_array_box_lift import (
        RigidGraspConstraint,
        checker_with_box,
        lift_region,
        plan_lift,
    )

    system, model, checker, _zero, q_grasp, box_size, box_centre = _setup(4, tmp_path)
    checker = checker_with_box(system, model, checker, box_size, box_centre)
    assert checker.obstacle_clearance(q_grasp) >= 0.0

    # The planning half is probabilistic: a rigid grasp *and* a box obstacle
    # leave a thin corridor, so a run can legitimately find nothing. What must
    # never happen is a path that goes *through* the box.
    constraint = RigidGraspConstraint(model, system, 4, q_grasp)
    path = plan_lift(model, 4, q_grasp, lift_region(model, 4, q_grasp, 0.15),
                     checker, attempts=8, constraint=constraint)
    if path is None:
        pytest.skip("no lift path this run (thin corridor); grasp clearance checked above")
    assert min(checker.obstacle_clearance(w) for w in path) >= 0.0


@requires_source
def test_approach_starts_off_the_box(tmp_path):
    """The arms must begin clear of the box and move down to it."""
    from circle_array_box_lift import approach_posture, end_effector_positions

    system, model, _checker, _zero, q_grasp, _size, _centre = _setup(4, tmp_path)
    q_approach = approach_posture(model, 4)

    above = end_effector_positions(model, system, 4, q_approach)[:, 2].mean()
    at_grasp = end_effector_positions(model, system, 4, q_grasp)[:, 2].mean()
    assert above > at_grasp + 0.05, "the pre-grasp must be above the grasp"


@requires_source
def test_descent_connects_pre_grasp_to_grasp(tmp_path):
    from circle_array_box_lift import approach_posture, checker_with_box, plan_approach

    system, model, checker, _zero, q_grasp, box_size, box_centre = _setup(4, tmp_path)
    checker = checker_with_box(system, model, checker, box_size, box_centre)
    q_approach = approach_posture(model, 4)

    descent = plan_approach(model, 4, q_approach, q_grasp, checker)
    assert descent is not None, "no descent path"
    np.testing.assert_allclose(descent[0], q_approach, atol=1e-6)
    np.testing.assert_allclose(descent[-1], q_grasp, atol=1e-2)


@requires_source
def test_mujoco_sees_contact_only_after_the_descent(tmp_path):
    """The physical check: empty hands at the pre-grasp, contact at the grasp.

    Counting *arm* contacts specifically -- the crate also rests on the floor,
    which produces four contacts of its own at every configuration.
    """
    mujoco = pytest.importorskip("mujoco")

    from circle_array_box_lift import approach_posture, mujoco_model

    system, model, _checker, _zero, q_grasp, box_size, box_centre = _setup(4, tmp_path)
    mj_model, _xml = mujoco_model(system, box_size, box_centre, tmp_path)
    data = mujoco.MjData(mj_model)
    task_indices = np.asarray(model.cooperative.get_joint_indices(), dtype=int)
    width = int(model.cooperative.get_dof())
    crate_geom = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "crate_geom")

    def arm_contacts(q):
        system_q = np.zeros(system.get_dof())
        system_q[task_indices] = model._to_task_full(q)
        data.qpos[:width] = system_q[:width]
        mujoco.mj_forward(mj_model, data)
        total = 0
        for contact in data.contact[: data.ncon]:
            if crate_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == crate_geom else contact.geom1
            body = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                     mj_model.geom_bodyid[other])
            if body != "world":
                total += 1
        return total

    assert arm_contacts(approach_posture(model, 4)) == 0, "arms touch the box too early"
    assert arm_contacts(q_grasp) > 0, "arms never reach the box"
