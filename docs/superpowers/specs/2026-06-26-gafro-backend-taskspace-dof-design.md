# Gafro backend: drive IK/planning off the task space's *controlled* DOF

**Date:** 2026-06-26
**Scope (two repos):**
1. `gafropy` — `src/gafropy/cpp/robots/TaskSpace.cpp` (expose the missing bindings) + wheel rebuild.
2. `pycbirrt` — `src/pycbirrt/backends/gafro.py` and `examples/franka_tsr_interactive.py`.

## Problem

Planning a `geodude.xml` `left_ur5e/endeffector_link` chain crashes in the gafro IK
solver:

```
ValueError: operands could not be broadcast together with shapes (7,) (6,)
  q = self._clamp_to_limits(q + dq)
```

### Root cause

gafro's C++ `TaskSpace` deliberately distinguishes two DOF counts, but the gafropy
binding only exposes one of them, so pycbirrt mixes them.

From the C++ source (`gafro/robot/task_space/TaskSpace.{hpp,hxx}`,
`SingleArmTaskSpace.hxx`):

- **`getDoF()`** = `dof_` = number of joints in the chain(s) the task space spans =
  **7** for this chain (includes the prismatic torso). Joint limits
  (`getJointLimitsMin/Max`) and `compute_ee_motor(position)` are all this
  **`dof_`-width** (7).
- **`getControlledDoF()`** = number of *controlled* joints = joints targeted by an
  actuator with control `group > 0` = **6** (the prismatic torso has no such
  actuator, so it is not controlled). Documented in the header as *"the width of
  the control Jacobian and command."*
- **`getControlledJoints()`** = the column indices (into the `dof_`-width vector)
  that are controlled.
- **`computeEEGeometricJacobian`** builds the full `dof_`-wide chain Jacobian and
  then selects exactly the `getControlledJoints()` columns, so it is
  **6 × controlledDoF** (6 columns), with columns ordered by `getControlledJoints()`.

So the 6-vs-7 split is **intended and explicit in gafro**: the geometric Jacobian
is the *control* Jacobian. The bug is that **the gafropy binding exposes only
`get_dof()` (=7) and omits `get_controlled_dof()` / `get_controlled_joints()`**, so
the pycbirrt backend sizes `q` and the joint-limit clamp from `get_dof()` (7) while
`dq = J.T @ ...` comes from the 6-column control Jacobian → `q + dq` is
`(7,) + (6,)`.

(There are genuinely three DOF concepts — `System::getDoF()` = 30,
`KinematicChain::getDoF()`, and `TaskSpace::getDoF()` = 7 with its separate
`getControlledDoF()` = 6. pycbirrt was reading the wrong one and the right one
wasn't bound.)

## Design principle

Use gafro's own API as the source of truth. Configurations, joint limits, and FK
are **`dof_`-width**; the planner's search space and the IK update are
**`controlledDoF`-width**; `getControlledJoints()` is the exact (possibly
non-contiguous) index map between them. No inference from Jacobian shape, no
"trailing joints" assumption.

The CBiRRT planner, the `RobotModel`/`IKSolver` Protocols, and `tsr/` are already
DOF-agnostic (they consume only `dof`, `joint_limits`, `forward_kinematics`,
`solve_valid`). They need **no changes**; they follow once the backend reports a
self-consistent DOF.

## Part A — gafropy bindings

In `src/gafropy/cpp/robots/TaskSpace.cpp`, in the `TaskSpace` base-class block
(both methods live on the base `gafro::TaskSpace<T>`), add:

```cpp
.def("get_controlled_dof", &GTaskSpace::getControlledDoF)
.def("get_controlled_joints", &GTaskSpace::getControlledJoints)  // vector<int>, indices into the dof_-width config
```

`<pybind11/stl.h>` is already included, so `std::vector<int>` converts to a Python
list automatically. Rebuild the wheel (per the gafropy venv memo: reinstall the
wheel, don't CMake-install over system paths). Verify from Python:
`get_controlled_dof() == 6`, `get_controlled_joints() == [1,2,3,4,5,6]`,
`len(get_joint_limits_min()) == 7`, and
`compute_ee_geometric_jacobian(zeros(7)).shape == (6, 6)`.

## Part B — pycbirrt gafro backend

### B1. Controlled-joint helper (shared by model + solver)

From a `SingleArmTaskSpace`, derive:

- `full_dof = get_dof()` (e.g. 7) — width of configs, limits, and FK input.
- `ctrl_idx = list(get_controlled_joints())` (e.g. `[1,2,3,4,5,6]`) — columns the
  geometric Jacobian drives, in Jacobian-column order.
- `ctrl_dof = get_controlled_dof()` (== `len(ctrl_idx)`, e.g. 6) — the DOF the
  planner plans in.
- Controlled joint limits = `full_limits[ctrl_idx]` for min and max (fancy-indexed
  in `ctrl_idx` order so they line up with Jacobian columns and `dq`).

Converters between the two widths:

- `to_full(q_ctrl, base_full)` → copy `base_full` (length `full_dof`), then
  `full[ctrl_idx] = q_ctrl`. `base_full` carries the held values of the
  non-controlled joints.
- `to_ctrl(q_full)` → `q_full[ctrl_idx]`.

`compute_ee_motor` and `compute_ee_geometric_jacobian` are always called with a
`full_dof` vector. Fast path: when `ctrl_dof == full_dof` and
`ctrl_idx == range(full_dof)` (every revolute arm today), both converters are
identities and behavior is unchanged.

### B2. `GafroIKSolver` — iterate in controlled space

- `self._dof = ctrl_dof`; `self.joint_limits` = controlled limits.
- The non-controlled joints are **held fixed** at the value implied by `q_init`:
  if `q_init` is `full_dof`-length, its values seed `base_full` (so the torso stays
  where the caller put it); if `q_init` is `ctrl_dof`-length or `None`, `base_full`
  defaults to the full-limit midpoint (torso at mid-range) and `q` starts from the
  controlled midpoint.
- Each iteration: `q_full = to_full(q, base_full)`; `current =
  compute_ee_motor(q_full)`; `error = log(target · current⁻¹)` (6-vec, unchanged);
  `J = compute_ee_geometric_jacobian(q_full)` (6 × ctrl_dof); damped-LS `dq` is
  length `ctrl_dof`; `q = clamp(q + dq)` now matches. Returns `ctrl_dof`-length
  configs.
- `solve_valid` limit-checks against controlled limits.

### B3. `GafroRobotModel` — consistent adapter (kept)

Kept as the `RobotModel` adapter (the example uses `.system` / `.mesh_root` /
`.manipulator` for the viewer). Made consistent via the same helper:

- `.dof` → `ctrl_dof`; `.joint_limits` → controlled limits.
- `.forward_kinematics(q)` accepts a `ctrl_dof`-length `q`, expands via
  `to_full(q, base_full)` with the held non-controlled joints, returns the `Motor`.
  The model's `base_full` defaults to the full-limit midpoint (documented), so the
  torso has a defined pose for FK/viz.

### B4. Example (`franka_tsr_interactive.py`)

- `START_Q` and `angular_joints` sized to `robot.dof` (now `ctrl_dof`); `START_Q`
  is the controlled-joint start.
- `_pad(q, system.get_dof())` still expands to the full **System** width for the
  viewer; it now receives a `ctrl_dof`-length `q`. Where the viewer needs the
  arm's pose it goes through the model's FK (full-chain reconstruction); the raw
  `robot_viz.update` keeps zero-padding against system DOF as today.

## Error handling

- The helper asserts `ctrl_dof == len(ctrl_idx)` and `max(ctrl_idx) < full_dof`;
  mismatch raises a clear `ValueError` (guards against a stale/partial binding).
- `ctrl_dof == full_dof` with identity `ctrl_idx` is an explicit fast path, so
  existing revolute-arm robots are byte-for-byte unaffected.

## Testing

1. **Binding smoke test (gafropy):** `get_controlled_dof() == 6`,
   `get_controlled_joints() == [1,2,3,4,5,6]`, geometric Jacobian is `(6,6)`,
   joint limits length 7.
2. **Regression (the crash):** plan the `geodude.xml` `left_ur5e/endeffector_link`
   chain to a TSR (`--no-viz`); assert no broadcast error and a path of
   `ctrl_dof`-width waypoints.
3. **DOF consistency:** `solver._dof == model.dof == get_controlled_dof() ==
   geometric_jacobian.shape[1]`; controlled-limit length matches.
4. **`J @ dq == error`:** one damped-LS step's `J @ dq` reproduces the CGA log
   error twist (guards the convention).
5. **No regression for revolute arms:** a plain 6/7-DOF arm where
   `ctrl_idx == range(dof)` solves IK and plans with identity mapping.
6. **Fixed-joint hold:** after IK, `to_full(result, base_full)`'s non-controlled
   entries equal `base_full` (the torso did not move).

## Out of scope

- No changes to `planner.py`, the Protocols, `tsr/`, or other backends
  (`mujoco.py`, `eaik.py`).
- No gafro C++ change — `getControlledDoF`/`getControlledJoints` already exist; we
  only bind them.
- No task-space-as-explicit-API-parameter surface across pycbirrt/tsr.
