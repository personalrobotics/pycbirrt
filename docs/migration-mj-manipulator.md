# Migrating mj_manipulator onto the native planning boundary

Written for the mj_manipulator maintainers, ahead of personalrobotics/mj_manipulator#174 and #175.

pycbirrt 1.7 offers one boundary for planning against a live MuJoCo world: an
owned scene, an immutable snapshot, a collision checker with mj_manipulator's
contact policy implemented in C++, and one call that runs the whole solve
natively. This guide maps what `Arm.create_planner` builds today onto that
boundary, says what pycbirrt does not take over, and states the rules a
caller must keep.

## The boundary

| Today, in `Arm.create_planner` | On the boundary |
|---|---|
| `self.env.fork()` for an isolated `MjData` | `NativeScene.from_model(model, joint_names, extra_arm_bodies)`: an `mjModel` pycbirrt owns, built once from the compiled model's MJB bytes and cached by their hash; nothing borrows the live model. |
| `ContextRobotModel(model, data, ...)` | `MuJoCoRobotModel(model, data, ee_site, joint_names)`, or nothing: `plan_native` builds it. Its forward kinematics must agree with the SSIK model's; lowering checks that at the start configurations and refuses with a reason if they differ by more than 1e-6. |
| `CollisionChecker(model, data, joint_names, grasped_objects=..., attachments=..., extra_arm_body_names=...)` in snapshot mode | `Snapshot.capture(scene, data, attachments={object: (gripper_body, T_gripper_object)})` on the owner thread, then `NativeCollisionChecker(scene, snapshot)`. The gripper-base rule (`<prefix>/base` when it exists, else the attachment body) is resolved to body ids at capture. |
| `continuous_joints(model, joint_names)` | Unchanged: `CBiRRTConfig(angular_joints=...)`. Unlimited joints are reported as ±∞ by both robot models and the space demands the declaration. |
| `CBiRRT(robot, ik, collision, config).plan(...)` | `plan_native(model, data, joint_names, ik=..., start=..., goal_tsrs=..., attachments=..., config=..., seed=...)`, or the same `CBiRRT(..., backend="native")` with a `NativeCollisionChecker` as the validator. |

The IK is `SSIKSolver(ssik.Manipulator.from_mjcf(ur5e_xml, base="world",
ee="wrist_3_link"), T_ee=site_offset_in_body(model, "attachment_site"))`,
built from the same MJCF as the MuJoCo model.

## What the result carries

`PlanResult.backend` is `"native"` or `"python"`; `backend_reasons` says why
Python was chosen under `backend="auto"`, the default since pycbirrt 2.0 (it
was `"python"` in 1.x; pass it explicitly to keep the reference). `PlanResult.provenance` records the
dependency versions, the scene's model signature and MJB hash, the snapshot
hash, and the SSIK family. `PlanResult.stats` breaks the cost down by
component. Zero Python functions run during a native solve; the reference
artifact records that count for the UR5e cases.

## What stays in mj_manipulator

Retiming, execution, visualization, grasp verification, recovery, and the
decision to replan. pycbirrt plans a geometric path against a snapshot and
returns it; it never touches the live `MjData` after capture (mj_manipulator#174's
first criterion), and it does not own the simulator or its threads.

## Rules for the caller

1. **Capture on the owner thread.** `Snapshot.capture` reads `qpos`, mocap
   poses, and computes attachment transforms from `data`; it must see a
   consistent world. After capture the solve is independent of `data`.
2. **Rebuild the scene after a structural change.** `NativeScene.from_model`
   caches by the MJB hash, so any change to the model, structural or numeric,
   yields a new scene on the next call; a cached scene never goes stale
   silently. `mjModel.signature` alone is not a safe key (it hashes structure
   only).
3. **Revalidate before executing.** The path was valid against the snapshot.
   Capture a fresh snapshot before execution and compare `sha256`; if it
   differs, run the path (or at least its executable prefix) through a
   `NativeCollisionChecker` on the fresh snapshot, and replan if it fails.
   The same policy that planned it validates it.
4. **One validator per solve.** Lowering creates a fresh validator with its
   own `mjData` for every solve; two solves in parallel never share one. Do
   not hand one `NativeCollisionChecker` to two threads' Python code either.
5. **Pin `mujoco==3.14.0`** while pycbirrt 1.7 is the version in use; the
   native scene refuses another MuJoCo with a message naming the three
   versions it sees.
6. **Unsupported means Python.** TSR chains, IK other than SSIK on a verified
   family, and any Python validator make `backend="native"` raise
   `NativeUnsupported` listing every blocker, and `backend="auto"` (or
   `plan_native(..., fallback=True)`) select the Python planner and record
   the reasons. The semantics are the same; the parity gate checks that.

## Downstream milestone

mj_manipulator#175 introduces the backend boundary in `Arm`, #174 bridges
snapshots and live-world revalidation, and #173 is the adoption artifact:
zero callbacks, independent validation, agreement with today's Python path
on a corpus, repeatable seeds, and full provenance. The corpus that gates
pycbirrt's validator against mj_manipulator's checker already exists
(`tests/reference/mujoco_collision_corpus.json`).
