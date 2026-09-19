# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""SSIK backend: adapter contract (with a fake solver) and SSIK integration (#63).

The adapter tests run without SSIK installed by stubbing the ``ssik`` module.
The integration tests skip unless the optional dependency is present.
"""

import importlib
import sys
import types

import numpy as np
import pytest

from pycbirrt.space import JointSpace


class FakeSolution:
    def __init__(self, q):
        self.q = q


class FakeSSIK:
    """Records solve kwargs and returns preset solutions."""

    def __init__(self, solutions):
        self.solutions = solutions
        self.calls = []

    def solve(self, T_target, **kwargs):
        self.calls.append((np.array(T_target), dict(kwargs)))
        return list(self.solutions)

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = np.asarray(q)[:3]
        return T


@pytest.fixture
def adapter_module(monkeypatch):
    """Import pycbirrt.backends.ssik with a stub ``ssik`` module if the real one is absent."""
    if "ssik" not in sys.modules:
        try:
            import ssik  # noqa: F401
        except ImportError:
            monkeypatch.setitem(sys.modules, "ssik", types.ModuleType("ssik"))
    import pycbirrt.backends.ssik as mod

    return importlib.reload(mod)


class TestAdapterContract:
    def test_q_init_is_forwarded_as_q_seed_with_limits_and_windings(self, adapter_module):
        fake = FakeSSIK([FakeSolution(np.zeros(6))])
        ik = adapter_module.SSIKSolver(fake)
        seed = np.arange(6, dtype=float)
        ik.solve(np.eye(4), q_init=seed)
        _, kwargs = fake.calls[-1]
        assert np.array_equal(kwargs["q_seed"], seed)
        assert kwargs["respect_limits"] is True
        assert kwargs["enumerate_windings"] is True
        assert "max_solutions" not in kwargs

    def test_no_seed_when_q_init_absent(self, adapter_module):
        fake = FakeSSIK([])
        adapter_module.SSIKSolver(fake).solve(np.eye(4))
        assert "q_seed" not in fake.calls[-1][1]

    def test_solutions_are_independent_float_arrays(self, adapter_module):
        shared = np.array([1, 2, 3, 4, 5, 6])  # int dtype, shared buffer
        fake = FakeSSIK([FakeSolution(shared), FakeSolution(shared)])
        out = adapter_module.SSIKSolver(fake).solve(np.eye(4))
        assert len(out) == 2
        for q in out:
            assert q.dtype == float and q.shape == (6,)
        out[0][0] = 99.0
        assert out[1][0] == 1.0 and shared[0] == 1

    def test_empty_stays_empty(self, adapter_module):
        assert adapter_module.SSIKSolver(FakeSSIK([])).solve(np.eye(4)) == []

    def test_all_solutions_preserved_no_cap_no_filter(self, adapter_module):
        sols = [FakeSolution(np.full(6, float(i))) for i in range(300)]
        out = adapter_module.SSIKSolver(FakeSSIK(sols)).solve(np.eye(4))
        assert len(out) == 300

    @pytest.mark.parametrize("bad", [np.array([np.nan] * 6), np.zeros((2, 3)), np.array([np.inf, 0, 0, 0, 0, 0])])
    def test_malformed_solution_raises(self, adapter_module, bad):
        ik = adapter_module.SSIKSolver(FakeSSIK([FakeSolution(np.zeros(6)), FakeSolution(bad)]))
        with pytest.raises(ValueError, match="malformed configuration at index 1"):
            ik.solve(np.eye(4))

    def test_bad_pose_shape_raises(self, adapter_module):
        with pytest.raises(ValueError, match="4x4"):
            adapter_module.SSIKSolver(FakeSSIK([])).solve(np.eye(3))

    def test_requires_a_solver_with_solve(self, adapter_module):
        with pytest.raises(TypeError):
            adapter_module.SSIKSolver(object())

    def test_frame_transforms_applied_to_target_and_fk(self, adapter_module):
        fake = FakeSSIK([FakeSolution(np.zeros(6))])
        T_base = np.eye(4)
        T_base[:3, 3] = [1.0, 0.0, 0.0]
        T_ee = np.eye(4)
        T_ee[:3, 3] = [0.0, 0.0, 0.5]
        ik = adapter_module.SSIKSolver(fake, T_base=T_base, T_ee=T_ee)
        pose = np.eye(4)
        pose[:3, 3] = [2.0, 3.0, 4.0]
        ik.solve(pose)
        target, _ = fake.calls[-1]
        assert np.allclose(target, np.linalg.inv(T_base) @ pose @ np.linalg.inv(T_ee))
        assert np.allclose(ik.fk(np.array([1.0, 2.0, 3.0, 0, 0, 0])), T_base @ fake.fk([1.0, 2.0, 3.0]) @ T_ee)

    @pytest.mark.parametrize("which", ["T_base", "T_ee"])
    def test_transforms_are_copied_so_caller_mutation_cannot_desync_inverse(self, adapter_module, which):
        fake = FakeSSIK([FakeSolution(np.zeros(6))])
        T = np.eye(4)
        ik = adapter_module.SSIKSolver(fake, **{which: T})
        pose = np.eye(4)
        pose[:3, 3] = [2.0, 3.0, 4.0]
        q = np.array([1.0, 2.0, 3.0, 0, 0, 0])
        ik.solve(pose)
        target_before, _ = fake.calls[-1]
        fk_before = ik.fk(q)

        T[0, 3] = 1.0  # caller mutates their array after construction

        ik.solve(pose)
        target_after, _ = fake.calls[-1]
        assert np.array_equal(target_after, target_before)
        assert np.array_equal(ik.fk(q), fk_before)
        stored = getattr(ik, which)
        assert stored[0, 3] == 0.0 and not stored.flags.writeable
        inv = ik._T_base_inv if which == "T_base" else ik._T_ee_inv
        assert np.allclose(stored @ inv, np.eye(4))

    def test_stored_transform_is_read_only(self, adapter_module):
        ik = adapter_module.SSIKSolver(FakeSSIK([]), T_base=np.eye(4))
        with pytest.raises(ValueError):
            ik.T_base[0, 3] = 5.0

    def test_bad_transform_rejected(self, adapter_module):
        with pytest.raises(ValueError, match="T_ee"):
            adapter_module.SSIKSolver(FakeSSIK([]), T_ee=np.eye(3))

    def test_no_collision_checker_accepted(self, adapter_module):
        with pytest.raises(TypeError):
            adapter_module.SSIKSolver(FakeSSIK([]), collision_checker=object())


class TestImportWithoutSSIK:
    def test_core_import_does_not_pull_in_ssik(self):
        """Importing pycbirrt never imports ssik or the backend module (checked in a fresh interpreter)."""
        import subprocess

        code = (
            "import sys, pycbirrt; "
            "assert 'ssik' not in sys.modules, 'ssik imported'; "
            "assert 'pycbirrt.backends.ssik' not in sys.modules, 'backend imported'"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

    def test_actionable_message_when_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ssik", None)  # makes `import ssik` raise ImportError
        sys.modules.pop("pycbirrt.backends.ssik", None)
        with pytest.raises(ImportError, match=r'pip install "pycbirrt\[ssik\]"'):
            importlib.import_module("pycbirrt.backends.ssik")
        sys.modules.pop("pycbirrt.backends.ssik", None)


# ---------------------------------------------------------------------------
# Integration with the real SSIK
# ---------------------------------------------------------------------------

ssik = pytest.importorskip("ssik")


@pytest.fixture(scope="module")
def ur5e():
    from ssik.prebuilt import ur5e_ik

    from pycbirrt.backends.ssik import SSIKSolver

    return SSIKSolver(ur5e_ik), ur5e_ik


class TestSSIKIntegration:
    Q = np.array([0.1, -1.2, 1.0, -0.5, 0.3, 0.2])

    def test_fk_ik_round_trip_is_certified(self, ur5e):
        ik, raw = ur5e
        T = ik.fk(self.Q)
        sols = ik.solve(T)
        assert sols
        for q in sols:
            assert q.shape == (6,)
            assert np.allclose(ik.fk(q), T, atol=1e-9)

    def test_unseeded_solve_exposes_distinct_in_limit_windings(self, ur5e):
        ik, raw = ur5e
        T = ik.fk(self.Q)
        sols = ik.solve(T)
        no_windings = raw.solve(T, enumerate_windings=False)
        assert len(sols) > len(no_windings)
        # Joint 0 has limits ±2π: the branch containing Q appears with q0 ≈ 0.1 and q0 ≈ 0.1 - 2π
        q0s = sorted({round(float(q[0]), 3) for q in sols if np.allclose(q[1:], self.Q[1:], atol=1e-6)})
        assert q0s == [round(0.1 - 2 * np.pi, 3), 0.1]

    def test_seeded_solve_puts_seed_nearest_winding_first(self, ur5e):
        ik, _ = ur5e
        T = ik.fk(self.Q)
        seed = self.Q.copy()
        seed[0] = 0.1 - 2 * np.pi + 0.02  # near the other in-limit winding of joint 0
        sols = ik.solve(T, q_init=seed)
        assert abs(sols[0][0] - seed[0]) < 0.1
        assert np.allclose(ik.fk(sols[0]), T, atol=1e-9)

    def test_windings_are_distinct_members_of_the_joint_space(self, ur5e):
        ik, _ = ur5e
        space = JointSpace(np.full(6, -2 * np.pi), np.full(6, 2 * np.pi))
        space.upper[2], space.lower[2] = np.pi, -np.pi
        sols = ik.solve(ik.fk(self.Q))
        kept = [q for q in sols if space.contains(q)]
        assert len(kept) == len(sols)
        # Two windings of joint 0 are 2π apart under the bounded metric, not zero
        a = next(q for q in kept if abs(q[0] - 0.1) < 1e-6 and np.allclose(q[1:], self.Q[1:], atol=1e-6))
        b = next(q for q in kept if abs(q[0] - (0.1 - 2 * np.pi)) < 1e-6 and np.allclose(q[1:], self.Q[1:], atol=1e-6))
        assert space.distance(a, b) == pytest.approx(2 * np.pi)

    def test_continuous_joints_do_not_enumerate_windings(self):
        """A model with no limits treats every joint as continuous: one representative per branch."""
        from pycbirrt.backends.ssik import SSIKSolver

        dh_alpha = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
        dh_a = np.array([0, -0.425, -0.3922, 0, 0, 0])
        dh_d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
        arm = ssik.Manipulator.from_dh(dh_alpha, dh_a, dh_d, limits=None)
        ik = SSIKSolver(arm)
        sols = ik.solve(ik.fk(self.Q))
        assert 0 < len(sols) <= 8  # one representative per geometric branch, no winding family
        for a in sols:
            for b in sols:
                if a is not b:
                    diff = np.abs(a - b)
                    # No pair is the same branch shifted by whole turns on some joints
                    assert not np.all((diff < 1e-9) | (np.abs(diff - 2 * np.pi) < 1e-6)) or np.all(diff < 1e-9)

    def test_unreachable_pose_gives_empty(self, ur5e):
        ik, _ = ur5e
        far = np.eye(4)
        far[0, 3] = 5.0
        assert ik.solve(far) == []
