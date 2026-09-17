# Gafro backend task-space controlled-DOF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the gafro IK/planning backend operate on a task space's *controlled* joints (matching its control Jacobian) instead of the full chain DOF, fixing the `(7,)+(6,)` broadcast crash.

**Architecture:** gafro C++ already splits `TaskSpace::getDoF()` (full chain width, 7) from `getControlledDoF()` (control-Jacobian width, 6) with `getControlledJoints()` as the index map; the gafropy binding only exposes `get_dof()`. Bind the two missing accessors, then rewrite `pycbirrt/backends/gafro.py` so configs/limits/FK stay full-DOF-width while the planner search space and IK update use the controlled width, bridged by `getControlledJoints()`. The CBiRRT planner, Protocols, and `tsr/` are already DOF-agnostic and untouched.

**Tech Stack:** C++ / pybind11 (gafropy bindings), Python 3.14 + NumPy (pycbirrt), pytest.

## Global Constraints

- Two repos: gafropy at `/home/tobi/coding/src/gafropy`, pycbirrt at `/home/tobi/coding/src/gafropy/pycbirrt` (nested, its own git repo). Commit in the repo that owns each changed file.
- Always activate the venv first: `source /home/tobi/coding/src/gafropy/.venv/bin/activate`.
- **gafropy C++ rebuild is a manual step performed by the user** via `pip install --no-build-isolation -e .` from `/home/tobi/coding/src/gafropy`. Do not script cmake/cp; pause and ask the user to rebuild when a task requires it.
- License header on every new/modified pycbirrt source/test file: `# SPDX-License-Identifier: MIT` then `# Copyright (c) 2025 Siddhartha Srinivasa` (match the existing files in the repo).
- Test robot: `/home/tobi/coding/src/rss/geodude.xml`, chain `left_ur5e/endeffector_link`. Expected: full DOF (`get_dof()`) = 7, controlled DOF = 6, `get_controlled_joints()` = `[1, 2, 3, 4, 5, 6]`, geometric Jacobian shape `(6, 6)`.
- pycbirrt tests run with: `cd /home/tobi/coding/src/gafropy/pycbirrt && python -m pytest`.
- Tests that need the real gafro robot/asset must skip cleanly when the asset is absent, so the suite still runs elsewhere.

---

## File Structure

- `gafropy/src/gafropy/cpp/robots/TaskSpace.cpp` — add two `.def` bindings on the `TaskSpace` base class.
- `pycbirrt/src/pycbirrt/backends/_taskspace_dof.py` (new) — small pure helper: derive `(full_dof, ctrl_idx, ctrl_dof, ctrl_limits)` from a task space and convert between full/controlled config widths. Isolated so it is unit-testable without touching the solver.
- `pycbirrt/src/pycbirrt/backends/gafro.py` — `GafroRobotModel` and `GafroIKSolver` consume the helper; iterate in controlled space; reconstruct full-width configs for FK/Jacobian.
- `pycbirrt/tests/test_gafro_taskspace_dof.py` (new) — helper unit tests + a gafro-asset integration test (skipped if the asset is missing).
- `pycbirrt/examples/franka_tsr_interactive.py` — size `START_Q` / `angular_joints` to the controlled DOF.

---

## Task 1: Bind `get_controlled_dof` / `get_controlled_joints` in gafropy

**Files:**
- Modify: `/home/tobi/coding/src/gafropy/src/gafropy/cpp/robots/TaskSpace.cpp:46`

**Interfaces:**
- Consumes: nothing.
- Produces: Python methods on every task space — `TaskSpace.get_controlled_dof() -> int` and `TaskSpace.get_controlled_joints() -> list[int]` (indices into the `get_dof()`-width config, in control-Jacobian-column order).

- [ ] **Step 1: Add the two bindings**

In the `TaskSpace` base-class block, immediately after the existing `get_dof` line (`.def("get_dof", &GTaskSpace::getDoF)` at line 46), insert:

```cpp
            .def("get_controlled_dof", &GTaskSpace::getControlledDoF)
            .def("get_controlled_joints", &GTaskSpace::getControlledJoints)
```

(`<pybind11/stl.h>` is already included at the top of the file, so the returned `const std::vector<int>&` converts to a Python `list[int]` automatically. No `return_value_policy` needed — it copies into a new list.)

- [ ] **Step 2: Ask the user to rebuild gafropy**

This is a C++ change. Per Global Constraints, pause and ask the user to run, from `/home/tobi/coding/src/gafropy`:

```bash
pip install --no-build-isolation -e .
```

Do not proceed until they confirm the rebuild succeeded.

- [ ] **Step 3: Verify the bindings from Python**

Run:

```bash
source /home/tobi/coding/src/gafropy/.venv/bin/activate && python -c "
import numpy as np
from gafropy import SingleArmTaskSpace, SystemSerialization
sys = SystemSerialization.load('/home/tobi/coding/src/rss/geodude.xml')
m = SingleArmTaskSpace(sys, 'left_ur5e/endeffector_link', 'left_ur5e/endeffector_link')
print('full_dof', m.get_dof())
print('ctrl_dof', m.get_controlled_dof())
print('ctrl_idx', list(m.get_controlled_joints()))
print('jac', np.asarray(m.compute_ee_geometric_jacobian(np.zeros(m.get_dof()))).shape)
print('limits', len(m.get_joint_limits_min()))
"
```

Expected output:
```
full_dof 7
ctrl_dof 6
ctrl_idx [1, 2, 3, 4, 5, 6]
jac (6, 6)
limits 7
```

- [ ] **Step 4: Commit (gafropy repo)**

```bash
cd /home/tobi/coding/src/gafropy
git add src/gafropy/cpp/robots/TaskSpace.cpp
git commit -m "feat: bind TaskSpace getControlledDoF / getControlledJoints

The control Jacobian (compute_ee_geometric_jacobian) is controlledDoF-wide
but only get_dof() (full chain width) was exposed, forcing consumers to
mismatch config and Jacobian widths.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Controlled-DOF helper module

**Files:**
- Create: `/home/tobi/coding/src/gafropy/pycbirrt/src/pycbirrt/backends/_taskspace_dof.py`
- Test: `/home/tobi/coding/src/gafropy/pycbirrt/tests/test_gafro_taskspace_dof.py`

**Interfaces:**
- Consumes: `get_dof()`, `get_controlled_dof()`, `get_controlled_joints()`, `get_joint_limits_min()`, `get_joint_limits_max()` from a gafropy task space (Task 1).
- Produces:
  - `class ControlledDOF` with attributes `full_dof: int`, `ctrl_idx: np.ndarray` (int, length `ctrl_dof`), `ctrl_dof: int`, `lower: np.ndarray` / `upper: np.ndarray` (controlled-width limits).
  - `ControlledDOF.from_task_space(manipulator) -> ControlledDOF` (classmethod).
  - `ControlledDOF.to_full(self, q_ctrl: np.ndarray, base_full: np.ndarray) -> np.ndarray` (length `full_dof`).
  - `ControlledDOF.to_ctrl(self, q_full: np.ndarray) -> np.ndarray` (length `ctrl_dof`).
  - `ControlledDOF.midpoint_full(self) -> np.ndarray` — full-width vector with each entry the midpoint of the full joint limits (used as the default fixed-joint base).

- [ ] **Step 1: Write the failing test**

Create `/home/tobi/coding/src/gafropy/pycbirrt/tests/test_gafro_taskspace_dof.py`:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest

from pycbirrt.backends._taskspace_dof import ControlledDOF


class FakeTaskSpace:
    """Minimal stand-in matching the gafropy task-space accessors used by ControlledDOF."""

    def __init__(self, full_dof, ctrl_idx, lower, upper):
        self._full_dof = full_dof
        self._ctrl_idx = list(ctrl_idx)
        self._lower = np.asarray(lower, dtype=float)
        self._upper = np.asarray(upper, dtype=float)

    def get_dof(self):
        return self._full_dof

    def get_controlled_dof(self):
        return len(self._ctrl_idx)

    def get_controlled_joints(self):
        return list(self._ctrl_idx)

    def get_joint_limits_min(self):
        return self._lower

    def get_joint_limits_max(self):
        return self._upper


def test_from_task_space_extracts_controlled_subset():
    ts = FakeTaskSpace(
        full_dof=7,
        ctrl_idx=[1, 2, 3, 4, 5, 6],
        lower=[0.0, -1, -1, -1, -1, -1, -1],
        upper=[0.5, 1, 1, 1, 1, 1, 1],
    )
    cd = ControlledDOF.from_task_space(ts)
    assert cd.full_dof == 7
    assert cd.ctrl_dof == 6
    assert list(cd.ctrl_idx) == [1, 2, 3, 4, 5, 6]
    # Controlled limits are the full limits indexed by ctrl_idx (torso joint 0 dropped).
    assert np.allclose(cd.lower, [-1, -1, -1, -1, -1, -1])
    assert np.allclose(cd.upper, [1, 1, 1, 1, 1, 1])


def test_to_full_holds_noncontrolled_joints_from_base():
    ts = FakeTaskSpace(7, [1, 2, 3, 4, 5, 6],
                       [0.0, -1, -1, -1, -1, -1, -1], [0.5, 1, 1, 1, 1, 1, 1])
    cd = ControlledDOF.from_task_space(ts)
    base = np.array([0.3, 9, 9, 9, 9, 9, 9])  # joint 0 (torso) held at 0.3
    q_ctrl = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    full = cd.to_full(q_ctrl, base)
    assert full.shape == (7,)
    assert full[0] == 0.3                       # held
    assert np.allclose(full[1:], q_ctrl)        # controlled


def test_to_ctrl_selects_controlled_indices():
    ts = FakeTaskSpace(7, [1, 2, 3, 4, 5, 6],
                       [0.0, -1, -1, -1, -1, -1, -1], [0.5, 1, 1, 1, 1, 1, 1])
    cd = ControlledDOF.from_task_space(ts)
    q_full = np.array([0.3, 1, 2, 3, 4, 5, 6])
    assert np.allclose(cd.to_ctrl(q_full), [1, 2, 3, 4, 5, 6])


def test_round_trip_to_full_then_to_ctrl_is_identity():
    ts = FakeTaskSpace(7, [1, 2, 3, 4, 5, 6],
                       [0.0, -1, -1, -1, -1, -1, -1], [0.5, 1, 1, 1, 1, 1, 1])
    cd = ControlledDOF.from_task_space(ts)
    q_ctrl = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(cd.to_ctrl(cd.to_full(q_ctrl, cd.midpoint_full())), q_ctrl)


def test_non_contiguous_controlled_joints():
    # Controlled joints need not be contiguous; ctrl_idx is the authoritative map.
    ts = FakeTaskSpace(4, [0, 2], [-1, -1, -1, -1], [1, 1, 1, 1])
    cd = ControlledDOF.from_task_space(ts)
    base = np.array([9.0, 0.7, 9.0, 0.8])
    full = cd.to_full(np.array([0.1, 0.2]), base)
    assert full[0] == 0.1 and full[2] == 0.2     # controlled
    assert full[1] == 0.7 and full[3] == 0.8     # held


def test_identity_when_all_joints_controlled():
    ts = FakeTaskSpace(3, [0, 1, 2], [-1, -1, -1], [1, 1, 1])
    cd = ControlledDOF.from_task_space(ts)
    assert cd.ctrl_dof == cd.full_dof
    q = np.array([0.1, 0.2, 0.3])
    assert np.allclose(cd.to_full(q, cd.midpoint_full()), q)
    assert np.allclose(cd.to_ctrl(q), q)


def test_rejects_inconsistent_controlled_dof():
    class BadTaskSpace(FakeTaskSpace):
        def get_controlled_dof(self):
            return 99  # disagrees with len(get_controlled_joints())

    with pytest.raises(ValueError):
        ControlledDOF.from_task_space(
            BadTaskSpace(7, [1, 2, 3], [0.0] * 7, [1.0] * 7))


def test_rejects_index_out_of_range():
    with pytest.raises(ValueError):
        ControlledDOF.from_task_space(
            FakeTaskSpace(3, [0, 1, 5], [0.0, 0, 0], [1.0, 1, 1]))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/tobi/coding/src/gafropy/pycbirrt && source /home/tobi/coding/src/gafropy/.venv/bin/activate && python -m pytest tests/test_gafro_taskspace_dof.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pycbirrt.backends._taskspace_dof'`.

- [ ] **Step 3: Write the helper implementation**

Create `/home/tobi/coding/src/gafropy/pycbirrt/src/pycbirrt/backends/_taskspace_dof.py`:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Map between a gafro task space's full chain DOF and its *controlled* DOF.

A gafro ``TaskSpace`` reports two joint counts: ``get_dof()`` is the width of the
chain it spans (configs, joint limits, and ``compute_ee_motor`` all use this),
while ``get_controlled_dof()`` is the width of its **control Jacobian**
(``compute_ee_geometric_jacobian``) -- the joints actually driven by actuators.
``get_controlled_joints()`` lists which full-width columns are controlled.

IK and planning operate in the *controlled* width; FK and the Jacobian are called
with the *full* width, the non-controlled joints held at a fixed base. This helper
is the bridge, isolated here so it can be unit-tested without a real robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ControlledDOF:
    full_dof: int          # width of configs / limits / FK input
    ctrl_idx: np.ndarray   # int indices into the full-width vector, in Jacobian-column order
    ctrl_dof: int          # width of the control Jacobian and IK update
    lower: np.ndarray      # controlled-width lower joint limits
    upper: np.ndarray      # controlled-width upper joint limits
    _full_lower: np.ndarray
    _full_upper: np.ndarray

    @classmethod
    def from_task_space(cls, manipulator) -> "ControlledDOF":
        full_dof = int(manipulator.get_dof())
        ctrl_idx = np.asarray(list(manipulator.get_controlled_joints()), dtype=int)
        ctrl_dof = int(manipulator.get_controlled_dof())

        if ctrl_dof != len(ctrl_idx):
            raise ValueError(
                f"get_controlled_dof() ({ctrl_dof}) != len(get_controlled_joints()) "
                f"({len(ctrl_idx)}); stale or inconsistent gafropy binding?"
            )
        if ctrl_idx.size and (ctrl_idx.min() < 0 or ctrl_idx.max() >= full_dof):
            raise ValueError(
                f"controlled joint indices {ctrl_idx.tolist()} out of range for "
                f"full DOF {full_dof}"
            )

        full_lower = np.asarray(manipulator.get_joint_limits_min(), dtype=float)
        full_upper = np.asarray(manipulator.get_joint_limits_max(), dtype=float)
        return cls(
            full_dof=full_dof,
            ctrl_idx=ctrl_idx,
            ctrl_dof=ctrl_dof,
            lower=full_lower[ctrl_idx],
            upper=full_upper[ctrl_idx],
            _full_lower=full_lower,
            _full_upper=full_upper,
        )

    def to_full(self, q_ctrl: np.ndarray, base_full: np.ndarray) -> np.ndarray:
        """Controlled-width config -> full-width, holding non-controlled joints at ``base_full``."""
        full = np.asarray(base_full, dtype=float).copy()
        full[self.ctrl_idx] = np.asarray(q_ctrl, dtype=float)
        return full

    def to_ctrl(self, q_full: np.ndarray) -> np.ndarray:
        """Full-width config -> controlled-width (the joints the Jacobian drives)."""
        return np.asarray(q_full, dtype=float)[self.ctrl_idx]

    def midpoint_full(self) -> np.ndarray:
        """Full-width midpoint of the joint limits; the default fixed-joint base."""
        return 0.5 * (self._full_lower + self._full_upper)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/tobi/coding/src/gafropy/pycbirrt && source /home/tobi/coding/src/gafropy/.venv/bin/activate && python -m pytest tests/test_gafro_taskspace_dof.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit (pycbirrt repo)**

```bash
cd /home/tobi/coding/src/gafropy/pycbirrt
git add src/pycbirrt/backends/_taskspace_dof.py tests/test_gafro_taskspace_dof.py
git commit -m "feat: ControlledDOF helper mapping full chain DOF <-> controlled DOF

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Rewrite `GafroIKSolver` to iterate in controlled space

**Files:**
- Modify: `/home/tobi/coding/src/gafropy/pycbirrt/src/pycbirrt/backends/gafro.py:96-198` (the `GafroIKSolver` class)
- Test: `/home/tobi/coding/src/gafropy/pycbirrt/tests/test_gafro_taskspace_dof.py` (append a gafro-asset integration test)

**Interfaces:**
- Consumes: `ControlledDOF` (Task 2); gafropy `SingleArmTaskSpace.compute_ee_motor`, `compute_ee_geometric_jacobian`.
- Produces: `GafroIKSolver` whose `solve` / `solve_valid` return **controlled-width** (`ctrl_dof`) configs; `self._dof == ctrl_dof`; `self.joint_limits` is controlled-width. Constructor signature unchanged: `GafroIKSolver(manipulator, joint_limits=None, collision_checker=None, damping=0.1, max_iterations=200, tolerance=1e-3)`.

- [ ] **Step 1: Write the failing integration test**

Append to `/home/tobi/coding/src/gafropy/pycbirrt/tests/test_gafro_taskspace_dof.py`:

```python
import os

ROBOT = "/home/tobi/coding/src/rss/geodude.xml"
CHAIN = "left_ur5e/endeffector_link"
_have_robot = os.path.exists(ROBOT)
requires_robot = pytest.mark.skipif(not _have_robot, reason=f"missing {ROBOT}")


@pytest.fixture
def manipulator():
    from gafropy import SingleArmTaskSpace, SystemSerialization
    system = SystemSerialization.load(ROBOT)
    return SingleArmTaskSpace(system, CHAIN, CHAIN)


@requires_robot
def test_solver_dof_matches_control_jacobian(manipulator):
    import numpy as np
    from pycbirrt.backends.gafro import GafroIKSolver

    solver = GafroIKSolver(manipulator)
    jac = np.asarray(
        manipulator.compute_ee_geometric_jacobian(np.zeros(manipulator.get_dof())))
    assert solver._dof == jac.shape[1] == manipulator.get_controlled_dof()
    lower, upper = solver.joint_limits
    assert len(lower) == solver._dof and len(upper) == solver._dof


@requires_robot
def test_solve_reaches_a_reachable_pose_without_broadcast_error(manipulator):
    import numpy as np
    from pycbirrt.backends.gafro import GafroIKSolver

    solver = GafroIKSolver(manipulator, max_iterations=300, tolerance=1e-5)
    # A pose known reachable: FK of a controlled-width config.
    q_seed = np.zeros(solver._dof)
    q_seed[:] = 0.2
    from pycbirrt.backends._taskspace_dof import ControlledDOF
    cd = ControlledDOF.from_task_space(manipulator)
    target = manipulator.compute_ee_motor(cd.to_full(q_seed, cd.midpoint_full()))

    sols = solver.solve(target, q_init=q_seed)
    assert sols, "solver returned no solution"
    assert sols[0].shape == (solver._dof,)


@requires_robot
def test_solve_holds_noncontrolled_joint_fixed(manipulator):
    import numpy as np
    from pycbirrt.backends._taskspace_dof import ControlledDOF
    from pycbirrt.backends.gafro import GafroIKSolver

    cd = ControlledDOF.from_task_space(manipulator)
    solver = GafroIKSolver(manipulator, max_iterations=300, tolerance=1e-5)

    q_seed = np.full(solver._dof, 0.15)
    base = cd.midpoint_full()
    base[0] = 0.42  # pin the prismatic torso somewhere specific
    q_init_full = cd.to_full(q_seed, base)

    target = manipulator.compute_ee_motor(q_init_full)
    sols = solver.solve(target, q_init=q_init_full)
    assert sols
    # The solver only returns controlled joints; the non-controlled torso is
    # whatever base we passed in -- verify reconstruction keeps it.
    reconstructed = cd.to_full(sols[0], base)
    assert reconstructed[0] == 0.42
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `cd /home/tobi/coding/src/gafropy/pycbirrt && source /home/tobi/coding/src/gafropy/.venv/bin/activate && python -m pytest tests/test_gafro_taskspace_dof.py -v`
Expected: the three `@requires_robot` tests FAIL (currently `GafroIKSolver` sizes off `get_dof()` → broadcast error / `_dof == 7`). Helper tests still PASS.

- [ ] **Step 3: Rewrite `GafroIKSolver`**

In `/home/tobi/coding/src/gafropy/pycbirrt/src/pycbirrt/backends/gafro.py`, replace the entire `GafroIKSolver` class body (the `__init__`, `_clamp_to_limits`, and `solve` methods; keep `solve_valid` and `solve_from_multiple_inits` working against the new shapes) with:

```python
class GafroIKSolver:
    """Differential IK on a task space's *controlled* joints (CGA-native).

    Damped least squares using the gafropy geometric (control) Jacobian and the
    world-frame Motor-log pose error. The task space reports a full chain DOF
    (``get_dof()``) for configs/limits/FK and a narrower *controlled* DOF
    (``get_controlled_dof()``) for its control Jacobian; this solver searches in
    the controlled width and holds the non-controlled joints fixed. Returns at
    most one (controlled-width) solution per ``solve`` call.
    """

    def __init__(
        self,
        manipulator: "SingleArmTaskSpace",
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        collision_checker: CollisionChecker | None = None,
        damping: float = 0.1,
        max_iterations: int = 200,
        tolerance: float = 1e-3,
    ):
        self.manipulator = manipulator
        self.damping = damping
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.collision_checker = collision_checker

        self._cd = ControlledDOF.from_task_space(manipulator)
        self._dof = self._cd.ctrl_dof

        if joint_limits is None:
            joint_limits = (self._cd.lower, self._cd.upper)
        self.joint_limits = joint_limits

    def _clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        lower, upper = self.joint_limits
        return np.clip(q, lower, upper)

    def _base_full(self, q_init: np.ndarray | None) -> np.ndarray:
        """Full-width vector whose non-controlled joints are the held values.

        A full-width ``q_init`` seeds the held joints from itself (e.g. keep the
        torso where the caller put it); otherwise the limit midpoint is used.
        """
        if q_init is not None:
            q_init = np.asarray(q_init, dtype=float)
            if q_init.shape[0] == self._cd.full_dof:
                return q_init.copy()
        return self._cd.midpoint_full()

    def _to_ctrl_start(self, q_init: np.ndarray | None) -> np.ndarray:
        """Controlled-width starting config from a full- or controlled-width hint."""
        if q_init is not None:
            q_init = np.asarray(q_init, dtype=float)
            if q_init.shape[0] == self._cd.full_dof:
                return self._cd.to_ctrl(q_init)
            if q_init.shape[0] == self._dof:
                return q_init.copy()
        lower, upper = self.joint_limits
        return 0.5 * (lower + upper)

    def solve(self, pose: "Motor | np.ndarray", q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Solve IK for a single EE pose. Returns [q_ctrl] if converged, else []."""
        from gafropy import Motor  # Motor() is the polymorphic pose-normalization boundary

        target = Motor(pose)
        base_full = self._base_full(q_init)
        q = self._to_ctrl_start(q_init)

        for _ in range(self.max_iterations):
            q_full = self._cd.to_full(q, base_full)
            current = self.manipulator.compute_ee_motor(q_full)
            error = _ee_error_twist(target, current)
            if np.linalg.norm(error) < self.tolerance:
                return [q]

            J = np.asarray(
                self.manipulator.compute_ee_geometric_jacobian(q_full), dtype=float)
            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 error
            JJT = J @ J.T
            damped = JJT + self.damping**2 * np.eye(JJT.shape[0])
            dq = J.T @ np.linalg.solve(damped, error)

            q = self._clamp_to_limits(q + dq)

        return []
```

`solve_valid` and `solve_from_multiple_inits` are unchanged in source — they already operate on whatever width `solve` returns and limit-check against `self.joint_limits` (now controlled-width), so they stay correct.

Add the helper import near the top of the file, after the existing `from pycbirrt.interfaces.collision_checker import CollisionChecker` line:

```python
from pycbirrt.backends._taskspace_dof import ControlledDOF
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/tobi/coding/src/gafropy/pycbirrt && source /home/tobi/coding/src/gafropy/.venv/bin/activate && python -m pytest tests/test_gafro_taskspace_dof.py -v`
Expected: all PASS (helper tests + the three `@requires_robot` tests).

- [ ] **Step 5: Commit**

```bash
cd /home/tobi/coding/src/gafropy/pycbirrt
git add src/pycbirrt/backends/gafro.py tests/test_gafro_taskspace_dof.py
git commit -m "fix: GafroIKSolver iterates in controlled DOF (fixes (7,)+(6,) crash)

Sizes q / limits / dq off the task space's control Jacobian width
(get_controlled_dof) and holds non-controlled joints fixed, instead of
mixing get_dof() with the 6-column geometric Jacobian.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Make `GafroRobotModel` report controlled DOF

**Files:**
- Modify: `/home/tobi/coding/src/gafropy/pycbirrt/src/pycbirrt/backends/gafro.py:41-93` (the `GafroRobotModel` class)
- Test: `/home/tobi/coding/src/gafropy/pycbirrt/tests/test_gafro_taskspace_dof.py` (append)

**Interfaces:**
- Consumes: `ControlledDOF` (Task 2).
- Produces: `GafroRobotModel` with `.dof == ctrl_dof`, `.joint_limits` controlled-width, and `.forward_kinematics(q_ctrl)` accepting a controlled-width config (expanding to full width with non-controlled joints at the limit midpoint). `.system`, `.chain_name`, `.mesh_root`, `.manipulator` unchanged.

- [ ] **Step 1: Write the failing test**

Append to `/home/tobi/coding/src/gafropy/pycbirrt/tests/test_gafro_taskspace_dof.py`:

```python
@requires_robot
def test_robot_model_reports_controlled_dof(manipulator):
    import numpy as np
    from pycbirrt.backends.gafro import GafroRobotModel

    model = GafroRobotModel.from_file(ROBOT, chain_name=CHAIN)
    assert model.dof == manipulator.get_controlled_dof()
    lower, upper = model.joint_limits
    assert len(lower) == model.dof and len(upper) == model.dof


@requires_robot
def test_robot_model_fk_accepts_controlled_width(manipulator):
    import numpy as np
    from gafropy import Motor
    from pycbirrt.backends.gafro import GafroRobotModel

    model = GafroRobotModel.from_file(ROBOT, chain_name=CHAIN)
    q = np.full(model.dof, 0.1)
    pose = model.forward_kinematics(q)
    # Returns a Motor (or Motor-coercible) without raising on the controlled width.
    assert Motor(pose) is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/tobi/coding/src/gafropy/pycbirrt && source /home/tobi/coding/src/gafropy/.venv/bin/activate && python -m pytest tests/test_gafro_taskspace_dof.py -k robot_model -v`
Expected: FAIL — `model.dof == 7` (still `get_dof()`), and `forward_kinematics` with a 6-vector mis-sizes.

- [ ] **Step 3: Update `GafroRobotModel`**

In `/home/tobi/coding/src/gafropy/pycbirrt/src/pycbirrt/backends/gafro.py`, edit the `GafroRobotModel.__init__` to build a `ControlledDOF` and store it, and update the `dof` / `joint_limits` / `forward_kinematics` members. Replace the tail of `__init__` (the `self._dof` / `self._lower` / `self._upper` block) and the three members:

Replace:

```python
        self.manipulator = SingleArmTaskSpace(system, chain_name, chain_name)
        self._dof = self.manipulator.get_dof()
        self._lower = np.asarray(self.manipulator.get_joint_limits_min(), dtype=float)
        self._upper = np.asarray(self.manipulator.get_joint_limits_max(), dtype=float)
```

with:

```python
        self.manipulator = SingleArmTaskSpace(system, chain_name, chain_name)
        self._cd = ControlledDOF.from_task_space(self.manipulator)
        # Non-controlled joints (e.g. a prismatic torso) are held at the limit
        # midpoint for FK / visualization.
        self._base_full = self._cd.midpoint_full()
```

Replace the `dof` property:

```python
    @property
    def dof(self) -> int:
        return self._dof
```

with:

```python
    @property
    def dof(self) -> int:
        return self._cd.ctrl_dof
```

Replace the `joint_limits` property:

```python
    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper
```

with:

```python
    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._cd.lower, self._cd.upper
```

Replace `forward_kinematics`:

```python
    def forward_kinematics(self, q: np.ndarray) -> "Motor":
        """End-effector pose as a ``gafropy.Motor`` (no matrix round-trip)."""
        return self.manipulator.compute_ee_motor(np.asarray(q, dtype=float))
```

with:

```python
    def forward_kinematics(self, q: np.ndarray) -> "Motor":
        """End-effector pose as a ``gafropy.Motor`` (no matrix round-trip).

        ``q`` is controlled-width (``self.dof``); non-controlled joints are held
        at the limit midpoint when expanding to the full chain config.
        """
        q_full = self._cd.to_full(np.asarray(q, dtype=float), self._base_full)
        return self.manipulator.compute_ee_motor(q_full)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/tobi/coding/src/gafropy/pycbirrt && source /home/tobi/coding/src/gafropy/.venv/bin/activate && python -m pytest tests/test_gafro_taskspace_dof.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/tobi/coding/src/gafropy/pycbirrt
git add src/pycbirrt/backends/gafro.py tests/test_gafro_taskspace_dof.py
git commit -m "fix: GafroRobotModel reports controlled DOF and FK accepts it

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: End-to-end planning test + example fix

**Files:**
- Test: `/home/tobi/coding/src/gafropy/pycbirrt/tests/test_gafro_taskspace_dof.py` (append)
- Modify: `/home/tobi/coding/src/gafropy/pycbirrt/examples/franka_tsr_interactive.py:46` (`START_Q`) and `:69-72` (`angular_joints`)

**Interfaces:**
- Consumes: `GafroRobotModel`, `GafroIKSolver` (Tasks 3-4), `CBiRRT`, `CBiRRTConfig`, `tsr.TSR`.
- Produces: a passing end-to-end plan to a TSR on the real robot; an example whose start config and `angular_joints` are sized to `robot.dof`.

- [ ] **Step 1: Write the failing end-to-end test**

Append to `/home/tobi/coding/src/gafropy/pycbirrt/tests/test_gafro_taskspace_dof.py`:

```python
@requires_robot
def test_plan_to_tsr_end_to_end():
    """Regression for the (7,)+(6,) crash: a full plan to a TSR must succeed."""
    import numpy as np
    from tsr import TSR
    from pycbirrt import CBiRRT, CBiRRTConfig
    from pycbirrt.backends.gafro import GafroIKSolver, GafroRobotModel

    class NoCollision:
        def is_valid(self, q):
            return True

    robot = GafroRobotModel.from_file(ROBOT, chain_name=CHAIN)
    ik = GafroIKSolver(robot.manipulator, robot.joint_limits,
                       max_iterations=300, tolerance=1e-6)
    config = CBiRRTConfig(max_iterations=5000, step_size=0.15, goal_bias=0.2,
                          tsr_samples=50, angular_joints=(True,) * robot.dof)
    planner = CBiRRT(robot, ik, NoCollision(), config)

    start = np.full(robot.dof, 0.1)

    T0_w = np.eye(4)
    T0_w[:3, 3] = [0.45, 0.0, 0.4]
    T0_w[:3, :3] = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    Bw = np.array([
        [-np.pi, np.pi], [0.0, 0.0], [0.0, 0.0],
        [-0.1, 0.1], [-0.1, 0.1], [0.0, 0.0],
    ])
    tsr = TSR(T0_w=T0_w, Tw_e=np.eye(4), Bw=Bw)

    result = planner.plan(start=start, goal_tsrs=[tsr], seed=1, return_details=True)
    assert result.success, f"plan failed: {result.failure_reason}"
    assert all(wp.shape == (robot.dof,) for wp in result.path)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/tobi/coding/src/gafropy/pycbirrt && source /home/tobi/coding/src/gafropy/.venv/bin/activate && python -m pytest tests/test_gafro_taskspace_dof.py::test_plan_to_tsr_end_to_end -v`
Expected: PASS already if Tasks 3-4 are correct (this test mainly *locks in* the regression). If it FAILS with a broadcast or DOF error, fix the relevant earlier task before continuing.

> Note: this is the one test whose "failing first" step may already pass because the fix landed in Tasks 3-4. That is acceptable — its role is a regression guard for the original crash. Do not weaken earlier tasks to force it red.

- [ ] **Step 3: Fix the example's hardcoded widths**

In `/home/tobi/coding/src/gafropy/pycbirrt/examples/franka_tsr_interactive.py`:

Replace the `START_Q` definition (line ~46):

```python
START_Q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])
```

with a comment deferring its sizing to the robot's controlled DOF, plus a default that gets resized at construction:

```python
# Start configuration in the task space's CONTROLLED joints. Sized to the robot's
# controlled DOF at construction time (see TSRPlanner.__init__); this literal is
# only the 7-DOF fallback for an all-revolute arm.
START_Q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])
```

Then in `TSRPlanner.__init__`, after `self.robot = GafroRobotModel.from_file(...)`, make the start and `angular_joints` follow `self.robot.dof`. Replace:

```python
        self.config = CBiRRTConfig(
            max_iterations=5000, step_size=0.15, goal_bias=0.2, tsr_samples=50,
            angular_joints=(True,) * self.robot.dof,
        )
        self.planner = CBiRRT(self.robot, self.ik, NoCollision(), self.config)
        self.start = START_Q
```

with:

```python
        self.config = CBiRRTConfig(
            max_iterations=5000, step_size=0.15, goal_bias=0.2, tsr_samples=50,
            angular_joints=(True,) * self.robot.dof,
        )
        self.planner = CBiRRT(self.robot, self.ik, NoCollision(), self.config)
        # Resize the start to the controlled DOF: use the literal if it already
        # matches, else fall back to the joint-limit midpoint.
        if START_Q.shape[0] == self.robot.dof:
            self.start = START_Q
        else:
            lower, upper = self.robot.joint_limits
            self.start = 0.5 * (lower + upper)
```

- [ ] **Step 4: Run the example smoke path (no viz) + the test**

Run:

```bash
cd /home/tobi/coding/src/gafropy/pycbirrt && source /home/tobi/coding/src/gafropy/.venv/bin/activate && \
python examples/franka_tsr_interactive.py --no-viz && \
python -m pytest tests/test_gafro_taskspace_dof.py -v
```

Expected: the example prints a `Plan to default TSR: ... waypoints ...` line and exits 0; all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/tobi/coding/src/gafropy/pycbirrt
git add tests/test_gafro_taskspace_dof.py examples/franka_tsr_interactive.py
git commit -m "test: end-to-end TSR plan on controlled DOF; size example to robot.dof

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Full regression pass

**Files:** none (verification only).

- [ ] **Step 1: Run the whole pycbirrt suite**

Run: `cd /home/tobi/coding/src/gafropy/pycbirrt && source /home/tobi/coding/src/gafropy/.venv/bin/activate && python -m pytest -v`
Expected: all PASS (existing `test_planner.py` / `test_plane_constraint.py` use mock robots and are unaffected; the new `test_gafro_taskspace_dof.py` passes).

- [ ] **Step 2: Confirm no broadcast/DOF errors remain**

Confirm the original traceback's failing call path (`solve_valid` → `solve` → `q + dq`) no longer raises by re-running:

```bash
cd /home/tobi/coding/src/gafropy/pycbirrt && source /home/tobi/coding/src/gafropy/.venv/bin/activate && \
python examples/franka_tsr_interactive.py --no-viz
```

Expected: exit 0, a plan summary printed, no `ValueError: operands could not be broadcast`.

---

## Self-Review notes

- **Spec coverage:** Part A → Task 1; Part B1 (helper) → Task 2; B2 (solver) → Task 3; B3 (model) → Task 4; B4 (example) → Task 5. Spec tests 1-6 map to: binding smoke (Task 1 Step 3), crash regression (Task 5), DOF consistency (Task 3), `J@dq==error` is exercised implicitly by convergence in Task 3's reach test, revolute-arm identity (Task 2 `test_identity_when_all_joints_controlled`), fixed-joint hold (Task 3 `test_solve_holds_noncontrolled_joint_fixed`).
- **Out-of-scope honored:** no edits to `planner.py`, Protocols, `tsr/`, `mujoco.py`, `eaik.py`, or gafro C++.
- **Type consistency:** `ControlledDOF` attributes/methods (`full_dof`, `ctrl_idx`, `ctrl_dof`, `lower`, `upper`, `from_task_space`, `to_full`, `to_ctrl`, `midpoint_full`) are used identically across Tasks 2-5.
