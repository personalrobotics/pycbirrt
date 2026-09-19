# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""MuJoCo differential IK: restart windows are topology-aware and reproducible (#69).

Needs mujoco only; models are tiny MJCF strings.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from pycbirrt.backends.mujoco import MuJoCoCollisionChecker, MuJoCoIKSolver  # noqa: E402

# A 3-DOF chain: limited hinge, unlimited hinge, limited slide; plus an unlimited slide on a fourth link.
XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="l1" pos="0 0 0.1">
      <joint name="h_lim" type="hinge" axis="0 0 1" limited="true" range="-1 1"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0"/>
      <body name="l2" pos="0.2 0 0">
        <joint name="h_free" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0"/>
        <body name="l3" pos="0.2 0 0">
          <joint name="s_lim" type="slide" axis="1 0 0" limited="true" range="0 0.3"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0"/>
          <body name="l4" pos="0.1 0 0">
            <joint name="s_free" type="slide" axis="0 0 1"/>
            <geom type="sphere" size="0.02"/>
            <site name="attachment_site" pos="0 0 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
JOINTS = ["h_lim", "h_free", "s_lim", "s_free"]


@pytest.fixture
def model_data():
    model = mujoco.MjModel.from_xml_string(XML)
    return model, mujoco.MjData(model)


def solver(model_data, **kw):
    model, data = model_data
    return MuJoCoIKSolver(model, data, "attachment_site", JOINTS, **kw)


class TestRestartBounds:
    def test_hinge_windows_are_one_turn_within_limits(self, model_data):
        ik = solver(model_data)
        lo, hi = ik._restart_bounds(np.array([0.5, 2.0, 0.1, 0.0]))
        assert (lo[0], hi[0]) == (-1.0, 1.0)  # limited hinge: one turn clipped to [-1, 1]
        assert lo[1] == pytest.approx(2.0 - np.pi) and hi[1] == pytest.approx(2.0 + np.pi)  # unlimited hinge

    def test_slide_windows_are_translational(self, model_data):
        ik = solver(model_data)
        lo, hi = ik._restart_bounds(np.array([0.0, 0.0, 0.1, 0.4]))
        assert (lo[2], hi[2]) == (0.0, 0.3)  # limited slide: its whole interval, no radians involved
        assert (lo[3], hi[3]) == pytest.approx((0.4 - 1.0, 0.4 + 1.0))  # unlimited slide: ±1 around the anchor

    def test_anchor_outside_limits_is_clamped_so_window_is_nonempty(self, model_data):
        ik = solver(model_data)
        lo, hi = ik._restart_bounds(np.array([5.0, 0.0, -2.0, 0.0]))
        assert np.all(lo <= hi)
        assert lo[0] <= 1.0 <= hi[0] and lo[2] <= 0.0 <= hi[2]

    def test_explicit_limits_disjoint_from_one_turn_do_not_raise(self, model_data):
        """The issue's reproduction: limits [4, 5] on every joint."""
        ik = solver(model_data, joint_limits=(np.full(4, 4.0), np.full(4, 5.0)), restarts=3)
        inits = []
        original = ik._solve_from

        def spy(pose, q0):
            inits.append(np.array(q0))
            return original(pose, q0)

        ik._solve_from = spy
        ik.solve(np.eye(4))  # must not raise
        assert len(inits) == 1 + 3
        for q0 in inits[1:]:
            assert np.all(q0 >= 4.0) and np.all(q0 <= 5.0)

    @pytest.mark.parametrize(
        "anchor", [np.zeros(4), np.array([0.9, -3.0, 0.29, 7.0]), np.array([-1.0, 10.0, 0.0, -7.0])]
    )
    def test_restarts_always_inside_limits(self, model_data, anchor):
        model, data = model_data
        ik = solver(model_data, restarts=5, seed=0)
        lo, hi = ik._restart_bounds(anchor)
        lower, upper = ik.joint_limits
        for _ in range(20):
            q0 = ik._rng.uniform(lo, hi)
            assert np.all(q0 >= lower) and np.all(q0 <= upper)


class TestReproducibility:
    def test_same_seed_same_restarts(self, model_data):
        model, _ = model_data
        a = MuJoCoIKSolver(model, mujoco.MjData(model), "attachment_site", JOINTS, seed=11, restarts=3)
        b = MuJoCoIKSolver(model, mujoco.MjData(model), "attachment_site", JOINTS, seed=11, restarts=3)
        seen = {"a": [], "b": []}
        for name, ik in (("a", a), ("b", b)):
            original = ik._solve_from
            ik._solve_from = lambda pose, q0, _o=original, _n=name: (seen[_n].append(np.array(q0)), _o(pose, q0))[1]
        target = np.eye(4)
        target[:3, 3] = [0.4, 0.0, 0.1]
        sa, sb = a.solve(target), b.solve(target)
        assert len(seen["a"]) == len(seen["b"]) == 4
        assert all(np.array_equal(x, y) for x, y in zip(seen["a"], seen["b"]))
        assert len(sa) == len(sb) and all(np.array_equal(x, y) for x, y in zip(sa, sb))

    def test_different_seeds_differ(self, model_data):
        model, _ = model_data
        a = MuJoCoIKSolver(model, mujoco.MjData(model), "attachment_site", JOINTS, seed=1, restarts=2)
        b = MuJoCoIKSolver(model, mujoco.MjData(model), "attachment_site", JOINTS, seed=2, restarts=2)
        lo, hi = a._restart_bounds(np.zeros(4))
        assert not np.array_equal(a._rng.uniform(lo, hi), b._rng.uniform(lo, hi))

    def test_seeded_solve_is_a_single_deterministic_attempt(self, model_data):
        ik = solver(model_data, restarts=5, seed=3)
        calls = []
        original = ik._solve_from
        ik._solve_from = lambda pose, q0: (calls.append(np.array(q0)), original(pose, q0))[1]
        target = np.eye(4)
        target[:3, 3] = [0.4, 0.0, 0.1]
        q_init = np.array([0.1, 0.2, 0.05, 0.0])
        ik.solve(target, q_init=q_init)
        assert len(calls) == 1 and np.array_equal(calls[0], q_init)
        state_before = ik._rng.bit_generator.state
        ik.solve(target, q_init=q_init)
        assert ik._rng.bit_generator.state == state_before  # no randomness consumed

    def test_examples_forward_seed_to_the_fallback(self, model_data):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
        import tsr_union_demo
        import ur5e_mujoco

        model, data = model_data
        collision = MuJoCoCollisionChecker(model, data, JOINTS)
        for mod in (ur5e_mujoco, tsr_union_demo):
            ik, name = mod.build_ik_solver(model, data, JOINTS, collision, Path("unused"), backend="mujoco", seed=7)
            assert name == "mujoco"
            assert ik._rng.bit_generator.state == np.random.default_rng(7).bit_generator.state
