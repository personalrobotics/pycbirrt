# Design: planning between sets

pycbirrt plans between sets. This document defines the objects the planner
works with, what each one means, and what each one is allowed to do. It is
the reference for the `PlanningProblem` API and for anyone adding a new kind
of set. The TSR-specific entry point, `plan(...)`, is one instantiation of
this design and is described at the end.

## The problem

Let $\mathcal{Q}$ be the configuration space. A planning problem names

- a **start set** $\mathcal{S} \subseteq \mathcal{Q}$,
- a **goal set** $\mathcal{G} \subseteq \mathcal{Q}$,
- a **path-admissible set** $\mathcal{C} \subseteq \mathcal{Q}$, and
- a **validator**, a predicate on $\mathcal{Q}$ (collision, in practice).

A solution is a path $\tau : [0, 1] \to \mathcal{Q}$ with $\tau(0) \in \mathcal{S}$,
$\tau(1) \in \mathcal{G}$, $\tau(t) \in \mathcal{C}$ for all $t$, and every
$\tau(t)$ valid. In code this is `PlanningProblem(space, start, goal,
validator, path_constraint)` and `CBiRRT.solve(problem)`.

These are **roles**. A set does not know which role it plays, and the same
set can serve as a goal in one problem and a path constraint in another.

## The state space

`JointSpace(lower, upper, angular_joints)` owns the geometry of
$\mathcal{Q}$: joint limits, the distance metric, the direction between two
configurations, straight-line interpolation, and uniform sampling. The planner
and every set that needs a metric share one instance.

A joint is **angular** only if it has no limits. Its distance wraps at $2\pi$
and its limit check always passes. A joint with limits is a bounded interval
however wide its range. The UR5e's $\pm 2\pi$ joints are bounded: a value and
that value plus one turn are different joint states, and moving between them
is a real full rotation. Marking such a joint angular lets the planner join
configurations a turn apart and the returned path spins the joint through
360° on execution.

## Sets: membership is the only requirement

A **state set** is anything with

```python
def contains(self, q: np.ndarray) -> bool
```

That is the whole definition. Fixed configurations, finite collections,
predicates, and TSR-induced sets are all state sets. The planner never
branches on a set's concrete type.

Leaves provided by `pycbirrt.sets`:

| Set | Members | Capabilities |
|---|---|---|
| `FiniteSet(configs, tolerance, metric)` | the listed configurations, within `tolerance` under `metric` | membership, distance, sampling |
| `PredicateSet(fn)` | whatever `fn` accepts | membership only |
| `EmptySet()` | nothing | membership only; finite |
| `TSRConfigurationSet(tsr, robot, ik, space, tolerance)` | $\{q : \mathrm{FK}(q) \in \mathrm{TSR}\}$ | membership, distance, sampling, projection |

## Capabilities: what a set can do, separately from what it is

Sampling, distance, and projection are optional. A set declares them by
providing the method; `supports(s, Capability)` asks.

- **`SetSampler.sample(rng) -> list[Sample]`** draws candidate members. One
  draw may yield several candidates, for example every IK branch of one
  sampled pose, or none. The caller validates them. A set never applies
  collision or other external filters to its own samples; that is the
  validator's job. This matters: which IK branch is collision-free is not
  knowable inside the set, and discarding branches before validation was a
  real bug.
- **`SetDistance.distance(q) -> float`** is a nonnegative geometric measure
  of how far `q` is from the set, before any tolerance is applied. For a
  leaf it is within the set's membership tolerance exactly when
  `contains(q)` holds; for composites it is only a summary (min for unions,
  max for intersections) with no membership threshold of its own.
- **`SetViolation.violation(q) -> float`** is nonnegative and exactly zero
  when `contains(q)` holds, on the set's own scale: for a TSR-induced set it
  is the TSR distance beyond the membership tolerance. Unions take the min
  and intersections the max, so `violation(q) == 0` agrees with membership
  at every level. Strategies that compare violations across children require
  them to be on a comparable scale, which the strategy cannot verify.
- **`SetProjector.project(q_previous, q_proposed) -> q | None`** moves a
  configuration onto the set. `q_previous` is where the extension started; a
  projector may use it to seed an iterative solve or to reject results that
  moved too far.

A **`Sample`** carries `q` and a `source` tuple: the sequence of choices that
produced it, outermost first. `AnyOf` prepends the index of the child it
chose; `FiniteSet` contributes the index of the member; other leaves
contribute nothing. This is how `PlanResult.start_source` and `goal_source`
report which alternative a path used.

## Composition

Two constructors, nesting freely:

- **`AnyOf(children, weights=None)`**: $q \in \bigcup_i C_i$.
- **`AllOf(children, projection=None, sampling=None)`**: $q \in \bigcap_i C_i$.

Membership is exact Boolean semantics in both cases, always available, and
grouping is preserved: $(L_1 \cap R_1) \cup (L_2 \cap R_2)$ describes two
matched pairings, while $(L_1 \cup L_2) \cap (R_1 \cup R_2)$ allows all four.

Capabilities of a composite are explicit, never implied:

| | `AnyOf` | `AllOf` |
|---|---|---|
| one child | delegates every capability to it | delegates every capability to it |
| distance | min over children, if all support it | max over children, if all support it (a geometric summary, not membership) |
| violation | min over children, if all support it | max over children, if all support it (zero iff member) |
| sampling | needs `weights`, the mixture policy; all children must sample | needs a named strategy, e.g. `RejectionSampling(source)` |
| projection | nearest successful child projection; all children must project | needs a named strategy, e.g. `MostViolatedProjection()` |

The single-child rule means a one-TSR constraint costs nothing and needs no
strategy. The multi-child `AllOf` rules are deliberate: there is no generic
way to sample or project onto an intersection, and the strategies that exist
are heuristics with names. Asking a composite for a capability it lacks
raises `UnsupportedCapability` with the reason.

Named strategies so far:

- `MostViolatedProjection(max_iters, progress_tolerance)`: repeatedly project
  onto the unsatisfied child with the largest `violation` until every child
  contains the point. A satisfied child is never selected. Progress is the
  lexicographic decrease of the descending-sorted violation profile, so
  clearing one of several equally violated children counts even though the
  maximum is unchanged; it gives up after a full sweep without progress,
  when a projector returns None or leaves the point unchanged, or at
  `max_iters`. Children must support violation and projection, and their
  violations must be comparable (homogeneous TSR sets are).
- `RejectionSampling(source)`: draw from one child and keep the candidates the
  others contain. Exact, but wasteful when the intersection is small.

Finite sets compose: a union of finite sets is finite, and an intersection
with a finite child is finite (enumerate that child, keep what the others
contain). `is_finite` and `members` implement this and the planner uses them
to collect roots.

No `Not`. Exclusion is the validator's job, and complements have no useful
sampler or projector.

## What the planner requires of each role

- **Every accepted configuration is a member of the joint space.** The
  first admissibility check for every root, sample, projected extension,
  and edge sample is `space.contains(q)`: a finite numeric array of shape
  `(dof,)` with every bounded joint inside its limits. A set, sampler, or
  projector that returns anything else is rejected with the reason
  "outside joint space", classified as invalid rather than as a collision.
  Nothing outside `problem.space` is ever stored in a tree. Concrete sets
  may filter limits early as an optimization, but correctness does not
  depend on it.
- **Start and goal** must be finite, sampleable, or both. Every explicit
  configuration embedded in the set is a candidate root: `seeds(s)` walks
  the expression and collects the members of finite sets, including those
  inside a union with a sampleable region, so mixture weights never decide
  whether a fixed configuration is a root. If the set is not finite and can
  sample, admissible candidates are added until `num_tree_roots` roots
  exist or the draw budget (`tsr_samples`) is spent, keeping at most
  `max_ik_per_pose` per draw for diversity and skipping candidates that
  repeat a seed. Bias sampling draws from any sampleable set.
- **Roots are taken from $\mathcal{S} \cap \mathcal{C}$ (and $\mathcal{G} \cap \mathcal{C}$) by
  rejection**: a candidate outside the path constraint or rejected by the
  validator is not a root. If no root survives, the planner raises the
  `All…Invalid` or `All…InCollision` exception for that role.
- **The path constraint** needs only membership. If it supports projection,
  every tree extension is projected onto it; otherwise an extension that
  leaves it is rejected. Every intermediate configuration on every edge is
  checked for membership and validity.
- **The validator** (state validity) is applied to every root and, through
  the default motion validator, to every configuration along every edge.
  State validity and motion validity are distinct interfaces:
  `CollisionChecker.is_valid(q)` for one configuration, `MotionValidator`
  for the motion between two.
- **Edges are the single local-motion boundary,** and it is explicit:
  `PlanningProblem.motion_validator` validates the local motion of every
  tree edge, the final connection between trees, and every shortcut, and
  returns the configurations to store (`LocalMotion`). The default,
  `DiscreteMotionValidator`, samples the straight segment every
  `edge_resolution` and requires every sample including the endpoint to be
  admissible, so validity is discrete, not continuous; choose the
  resolution against the thinnest obstacle you must not miss. A backend may
  supply continuous collision checking or a swept-volume check instead. A
  custom validator can only be stricter: the planner re-checks every
  configuration it returns for admissibility before storing it, and a
  motion that claims to reach must end exactly at the target.
- **Reached means connected.** Growth reports success only once the tree
  contains the exact target, added through a validated edge. Coming within
  `connection_tolerance` triggers that final exact edge; it never substitutes
  for it. Where the two trees meet, the connecting tree holds the other
  tree's configuration exactly and the join has no unchecked gap.
- **Shortcuts are validated edges that end at their target.** Smoothing
  accepts a shortcut only if its joint-space path length under the problem's
  space is shorter than the segment it replaces, as in the original CBiRRT;
  fewer waypoints is not the criterion. The first and last waypoints of a
  path are preserved exactly.

## Tolerances

Each has one meaning:

| `CBiRRTConfig` field | meaning |
|---|---|
| `membership_tolerance` | a configuration is in a TSR-induced set if its TSR distance is within this; projection is done when within it |
| `connection_tolerance` | tree growth counts as reaching its target within this joint-space distance |
| `edge_resolution` | spacing of validity checks along an edge (`None` means `step_size`) |
| `progress_tolerance` | tree growth stops when the distance to target shrinks by less than this |
| `projection_progress_tolerance` | projection gives up when the violation shrinks by less than this |

Membership tolerance belongs to the concrete set; the legacy lowering copies
the config value into the sets it builds. `tsr_tolerance` is a deprecated
alias that sets membership and connection tolerances together.

## The TSR instantiation

`plan(start, goal, goal_tsrs, start_tsrs, constraint_tsrs)` lowers its
arguments into a `PlanningProblem` (`pycbirrt.legacy`) and reproduces the
semantics the arguments always had:

- fixed configurations become a `FiniteSet` whose members are always roots
  and are never sampled;
- a list of start or goal TSRs becomes an `AnyOf` of `TSRConfigurationSet`s
  with weights proportional to TSR volume;
- a list of constraint TSRs becomes an `AllOf` with `MostViolatedProjection`,
  or the bare set when there is one TSR;
- `start_index` and `goal_index` are the last element of the root's
  provenance, which the lowering arranges to be the index into the config
  list or the TSR list.

`TSRConfigurationSet` samples poses from the TSR bounds with the planner's
random generator, solves IK, and returns every solution inside the joint
limits as candidates. Its projection moves the pose to the closest point of
the TSR, composed with the TSR's `T0_w` and `Tw_e` frames, and takes the IK
solution nearest the current configuration under the space metric.

## TSR chains

A TSR chain (see #7) couples several TSRs through kinematic composition: the
pose of each link constrains the next, and the chain as a whole defines
**one** set of end-effector poses. It is therefore one state set, induced
through forward kinematics like a single TSR, with its own sampler and
projector that walk the chain. It is **not** an `AllOf` of its constituent
TSRs: those TSRs constrain different frames, and their intersection in
configuration space is not the chain's induced set. Multi-link constraints
where each link has an independent TSR (an end-effector region and an elbow
clearance region) are a different object, and *those* compose as `AllOf`
with a joint projection strategy.
