# Changelog

All notable changes to pycbirrt. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/).

## [1.3.0] - 2026-09-19

The analytical IK backend moves from EAIK to SSIK. `plan(...)` and the
planner core are unchanged; the minor bump is for the new `pycbirrt[ssik]`
extra, the reduced `IKSolver` protocol, and the EAIK deprecation. Users of
the MuJoCo differential fallback should read the Fixed section: it was not
runnable from the examples in 1.2.0 and its unseeded behavior has changed.

### Added

- **SSIK backend** (`pycbirrt.backends.ssik.SSIKSolver`, extra
  `pycbirrt[ssik]`, requires `ssik>=6.0.1,<7`): enumerative analytical IK
  for 6R and 7R arms. The adapter wraps an `ssik.Manipulator` or a prebuilt
  artifact, forwards `q_init` as SSIK's seed, requests limit-respecting
  solutions with in-limit winding enumeration, applies no solution cap, does
  no collision checking, and accepts optional fixed `T_base` / `T_ee`
  transforms for frame conformance. Its `fk` lets you assert the frame
  contract against your `RobotModel` before planning (#63).
- `pycbirrt.backends.mujoco.site_offset_in_body(model, site)`: the fixed
  transform to pass as `T_ee` when SSIK is built from the same MJCF with the
  end-effector body as `ee`; the UR5e examples and integration test use it and
  match MuJoCo's forward kinematics to machine precision.

### Fixed

- The UR5e examples' MuJoCo differential-IK fallback crashed on the first
  IK update because the collision checker was passed in the `joint_limits`
  positional slot. Both examples now select the backend through
  `build_ik_solver(..., backend)` with a `--ik {auto,ssik,mujoco}` flag, and
  the fallback is exercised by tests even when SSIK is installed (#65).
- `MuJoCoIKSolver.solve` without a seed tries the current state and then
  `restarts` (default 3) random initial configurations within limits and
  returns every distinct converged solution. Previously it depended on
  whatever the shared MuJoCo state was last left in and converged on only
  about half of reachable poses, which made TSR goal sampling through the
  fallback unreliable.
- `MuJoCoIKSolver` restart windows are anchored at the current configuration
  and derived from each joint's MuJoCo type and limits: hinges sample one
  turn around the anchor intersected with their limits, limited slides their
  whole interval, unlimited slides ±1 around the anchor. The previous blanket
  intersection with [-π, π] raised for a valid interval such as [4, 5] and
  applied a revolute rule to prismatic joints. The examples forward their
  `--seed` to the fallback so runs are reproducible (#69).
- `SSIKSolver` copies `T_base` and `T_ee` at construction and stores them
  read-only, so mutating the caller's array can no longer desynchronize the
  stored transform from its cached inverse (#66).

### Changed

- The `IKSolver` protocol requires only `solve(pose, q_init)`. `solve_valid`
  is no longer part of the interface; the planner never called it, and joint
  limits and collision are the planner's responsibility. Existing backends
  keep it as a convenience.
- The UR5e examples prefer SSIK and fall back to MuJoCo differential IK.
- Because SSIK returns every in-limit winding, the TSR-induced set on
  joints wider than one turn is now complete; #36 is resolved in the IK
  backend, not in the planner.

### Deprecated

- `EAIKSolver` warns on construction. It, the `eaik` extra, and its
  documentation will be removed in pycbirrt 2.0.

## [1.2.0] - 2026-09-17

A hardening release after two adversarial reviews of the 1.1.0 set-based
planner (#42 to #47, #54 to #57). Every item below was reproduced on 1.1.0
and is guarded by tests written against the invariant it restores. The
`plan(...)` signature is unchanged; the minor bump is for the new
motion-validation API and the new set capabilities. Users of custom sets or
validators should read the Fixed section: paths, roots, and projection
results can differ from 1.1.0 where 1.1.0 was wrong.

### Added

- `JointSpace.contains(q)` and `why_invalid(q)`: the authoritative
  membership test for the joint space (shape, finiteness, limits) with a
  reason naming the offending joint (#43).
- `seeds(s)`: the explicit configurations embedded in any set expression,
  distinct from exhaustive `members(s)` (#42, #55).
- `SetViolation` capability: `violation(q)` is zero exactly when the set
  contains `q`; `TSRConfigurationSet` and `FiniteSet` implement it, and
  `AnyOf`/`AllOf` compose it (#44).
- `MotionContractError`, raised when a `MotionValidator` violates the
  `LocalMotion` contract (#54).
- `RestrictedMotionValidator(base, accepts)` and
  `CBiRRT.default_motion_validator(problem)` for explicit composition of a
  motion restriction with the default checks (#56).
- `PlanningProblem.motion_validator`: local-motion validation is an
  explicit, replaceable boundary. `MotionValidator.validate(q_from, q_to)`
  returns a `LocalMotion` (the validated configurations to store, and
  whether the target was reached), and every tree edge, the final
  connection between trees, and every shortcut go through it. The default
  `DiscreteMotionValidator` reproduces the discretized behavior at
  `edge_resolution`. A custom validator *replaces* the default and owns the
  motion's interior; the planner independently checks every configuration
  it returns before storing it. `RestrictedMotionValidator(base, accepts)`
  composes an extra restriction with a base validator (typically
  `planner.default_motion_validator(problem)`) to be stricter while keeping
  the default checks. Goal-tree edges are validated in the reverse of
  execution direction (#46, #56).

### Fixed

- The planner checks a `MotionValidator`'s whole `LocalMotion` before
  touching the tree. `reached=True` with no configurations on a nonzero
  motion, or ending anywhere but the exact target, raises the new
  `MotionContractError` (a validator bug, not a planning failure). A
  claimed success containing an inadmissible configuration is rejected
  whole; a partial result keeps its admissible prefix; a zero-length
  motion succeeds without adding a node whatever the payload (#54).
- `MostViolatedProjection` required the single largest violation to
  decrease after every projection, so an intersection with tied
  violations (two axes from a corner) was reported as stalled after the
  first child was satisfied. Progress is now the lexicographic decrease
  of the sorted violation profile; it stops after a full sweep without
  progress, when a projector makes no change, or at `max_iters` (#57).
- `members` of a finite intersection enumerates a child that is itself
  finite, not the first child that merely has explicit seeds, so an
  `AllOf` whose first seed-bearing child is a mixed union still
  enumerates exhaustively and can seed a search. `seeds` of a non-finite
  intersection now collects from every seed-bearing child, keeps what the
  whole intersection contains, and deduplicates equal configurations by
  first occurrence (#55).
- Membership in the ambient `JointSpace` (shape, finiteness, joint limits)
  is now the first admissibility check for every root, sample, projected
  extension, and edge sample. Out-of-limit or malformed fixed starts and
  goals are rejected before search with the `...Invalid` exceptions and a
  reason that names the joint; samplers and projectors that return
  out-of-space configurations can no longer get them into a tree (#43).
- Root sampling raises the `...InCollision` exception only when every
  rejected candidate was a collision; any other reason makes it `...Invalid`.
- Fixed configurations are always tree roots, including when a start or
  goal role also has sampled TSR alternatives. `plan(start=[q],
  start_tsrs=[...])` silently dropped `q` in 1.1.0 because the mixed union
  was not finite. New `seeds(s)` enumerates the explicit configurations
  embedded in any set expression; mixture weights now govern sampling
  only (#42).
- **Tree connection and shortcuts are validated to the exact target.**
  Growth reported success when merely within `connection_tolerance` of its
  target, leaving the final gap unchecked in the returned path, and shortcut
  smoothing snapped its last waypoint to the target without validating the
  segment (and could drop the shortcut's start when growth succeeded
  immediately). Now `reached` means the tree contains the exact target
  through a validated edge; the edge routine validates every sample
  including the endpoint; the join between trees carries no duplicate; and
  smoothing accepts a shortcut only when its joint-space length is shorter
  than the segment it replaces. Paths keep their first and last waypoints
  exactly (#47).
- `MostViolatedProjection` ranked children by raw `distance`, so a satisfied
  child with a large tolerance could outrank an unsatisfied one and the
  projection stalled on a feasible intersection. New `SetViolation`
  capability: `violation(q)` is zero exactly when the set contains `q`
  (`max(0, distance - tolerance)` for TSR and finite sets; min over a
  union, max over an intersection). The strategy now selects only among
  unsatisfied children by violation, and requires it of every child (#44).
- Nearest-neighbor selection uses the query's `PlanningProblem.space`, not
  the planner's construction-time space, so a direct `solve(problem)` with
  a different joint topology is internally consistent (#45).

## [1.1.0] - 2026-09-16

Additive and backward compatible. `plan(...)` keeps its signature and
semantics; every 1.0.0 test passes unchanged.

### Added

- **Planning between sets.** `CBiRRT.solve(problem)` takes a
  `PlanningProblem` made of a `JointSpace`, a start set, a goal set, a
  validator, and an optional path-admissible set. `plan(...)` now lowers its
  arguments into this representation. See `docs/design.md`.
- **State sets** (`pycbirrt.sets`): a membership-only `StateSet` protocol
  with optional `SetSampler`, `SetDistance`, and `SetProjector` capabilities;
  `FiniteSet`, `PredicateSet`, `EmptySet`; `AnyOf` (union) and `AllOf`
  (intersection) with arbitrary nesting and explicit capability rules; the
  named strategies `MostViolatedProjection` and `RejectionSampling`;
  `supports`, `is_finite`, `members`.
- **`TSRConfigurationSet`**, the configuration-space set a TSR induces
  through forward kinematics, with membership, distance, sampling, and
  IK-based projection. `tsr_weights` gives the volume-weighted mixture.
- **`JointSpace`**: joint limits, angular-aware metric and direction,
  interpolation, and uniform sampling in one object (`planner.space`).
- **Distinct tolerances** on `CBiRRTConfig`: `membership_tolerance`,
  `connection_tolerance`, `edge_resolution`, `progress_tolerance`,
  `projection_progress_tolerance`. Defaults reproduce 1.0.0 behavior.
- `PlanResult` gains `start_source` and `goal_source` (which alternative a
  path used, as a path through nested sets), `failure_reason`,
  `planning_time`, `tree_sizes`, and the search trees `tree_start` and
  `tree_goal` for inspection.
- `abort_fn` on `CBiRRTConfig` to stop planning early (#12).
- Failed root sampling reports why: IK unreachable, in collision, or
  constraint violated.
- Integration test on the UR5e in MuJoCo (skipped without the menagerie),
  hypothesis property tests for the set laws, and headless smoke tests for
  the examples.
- MIT license, SPDX headers, issue and PR templates.

### Changed

- **Sampling from TSRs is reproducible under a seed.** 1.0.0 used numpy's
  global random state for poses, so seeded plans were not repeatable.
- Goal and start bias apply to any sampleable set, including finite sets.
- With several constraint TSRs, projection converges each TSR before
  re-evaluating which is most violated. Single-TSR constraints are unchanged.
- Bias samples are collision-filtered by the validator rather than by
  `IKSolver.solve_valid`; only `solve` is required of a solver for sets.
- TSR-sampled goal configurations are collision-checked before becoming tree
  roots.
- The tsr package is imported only by `pycbirrt.tsr_set` and
  `pycbirrt.legacy`.

### Deprecated

- `CBiRRTConfig.tsr_tolerance`. Passing it sets `membership_tolerance` and
  `connection_tolerance` together and emits a `DeprecationWarning`; reading
  it returns `membership_tolerance`.

### Fixed

- **Constraint projection with non-identity TSR frames.** The closest
  in-bounds point was used as a world pose without composing `T0_w` and
  `Tw_e`, so projection failed whenever a constraint TSR had a translated or
  rotated frame (#28).
- **Do not mark bounded joints as angular.** The UR5e examples marked its
  ±2π joints angular, which let the planner join configurations a full turn
  apart and spin a joint through 360° on execution (#35). `angular_joints`
  is for genuinely unlimited joints only; the config comment now says so.
  If you copied the old example configuration, remove `angular_joints`.
- `examples/planar_arm.py` and `examples/multi_config_demo.py` crashed on a
  result field removed in March (#32).
- Collision checker state no longer goes stale between queries (#11).
- CI: the formatter version is pinned to the 0.15 series and a stale lock
  file that could never be honored was removed (#26).

### Known limitations

- On joints with a range wider than 2π, `TSRConfigurationSet` offers only
  the IK solver's representative of each branch, not its in-limit 2π twins,
  so a short route through the twin can be planned as a long one (#36).
- TSR chains are not yet supported (#7).

## [1.0.0] - 2026-02-02

Initial release: CBiRRT with TSR start, goal, and path constraints; MuJoCo
and EAIK backends; planar arm and UR5e examples.

[1.3.0]: https://github.com/personalrobotics/pycbirrt/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/personalrobotics/pycbirrt/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/personalrobotics/pycbirrt/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/personalrobotics/pycbirrt/releases/tag/v1.0.0
