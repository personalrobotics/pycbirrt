# Changelog

All notable changes to pycbirrt. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

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

[1.1.0]: https://github.com/personalrobotics/pycbirrt/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/personalrobotics/pycbirrt/releases/tag/v1.0.0
