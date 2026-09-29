# Native design: the SSCBiRRT C++20 contract

This document is the normative boundary for the native implementation of
pycbirrt (#82). It translates the Python design in [design.md](design.md)
into C++20 types without changing its semantics. The two backends are one
contract with two implementations; the only recorded difference is the
random-number engine, see [Concept map](#concept-map). Where the Python
reference needed a rule stated more sharply to make that true (finite
limits, config ranges, cancellation points, the free-space sampler), the
reference was changed first (#107, #108, #109, #110) and this document
describes it as it now behaves. Implementation begins only after this document
is reviewed and merged; #85 implements it, #92 gates it against the
reference artifact.

The reference for "what the planner does" is the Python implementation at
v1.4.0 and the artifact `tests/reference/python_reference.json`. This
document does not restate every rule in design.md; it says how each rule is
carried into C++ and what the C++ types promise.

## Scope of v1.5.0

The native core plans non-TSR problems: joint spaces with bounded and
angular joints, finite sets, `AnyOf` and `AllOf` of finite and native sets,
native state validators, native path constraints, a replaceable free-space
sampler, deadlines, cancellation, provenance, and the full search (roots, bidirectional growth, exact
connection, complete-edge validation, extraction, shortcutting, unwrapping).
The Python planner is unchanged and remains the reference and the fallback.

Not in this release: TSR-induced sets, IK, MuJoCo, any Python callback
inside the native solve, parallel search, a dynamic plugin ABI. Extensibility
is a source-level C++ API: users subclass the interfaces below and link
against `sscbirrt::core`.

## Dependency boundary

`sscbirrt::core` depends on the C++20 standard library and nothing else. It
does not depend on Python, pybind11, numpy, Eigen, sstsr, SSIK, MuJoCo,
OMPL, RoboPlan, or `mj_manipulator`. Configurations are `std::vector<double>`
and views are `std::span<const double>`; a consumer that uses Eigen converts
at the boundary. The reason is the standalone consumer criterion in #85: a
C++ project must be able to `find_package(sscbirrt)` and build with no
transitive dependency to resolve. Eigen may be adopted inside later targets
(pose regions in v1.6.0) without touching the core.

```
pycbirrt._native  (pybind11)   ──►  sscbirrt::core  ◄──  sscbirrt_tests, sscbirrt_consumer
                                          ▲
                        (v1.6.0) sscbirrt::tsr ─┘   (v1.7.0) sscbirrt::mujoco ─┘
```

Arrows point from dependent to dependency. Nothing points out of `core`.

## Vocabulary

| Name | Definition |
|---|---|
| `Config` | `std::vector<double>` of length `dof`; owned. |
| `ConfigView` | `std::span<const double>`; borrowed, valid for the call. |
| `Provenance` | `std::vector<int>`; the sequence of choices that produced a sample, outermost first. Empty for a leaf. |
| `Rng` | `std::mt19937_64`; owned by the solver, passed by reference to sets. |
| `Metric` | `std::function<double(ConfigView, ConfigView)>`; a set's notion of distance between configurations. |

## Joint space

```cpp
namespace sscbirrt {

class SpaceSampler {
public:
  virtual ~SpaceSampler() = default;
  virtual Config sample(Rng& rng) const = 0;            // a free-space target of length dof
};

class JointSpace final : public SpaceSampler {
public:
  JointSpace(std::vector<double> lower, std::vector<double> upper,
             std::vector<bool> angular = {});           // throws std::invalid_argument

  int dof() const;
  const std::vector<double>& lower() const;
  const std::vector<double>& upper() const;
  const std::vector<bool>& angular() const;             // all false if none

  bool contains(ConfigView q) const;                    // shape, finiteness, limits
  std::optional<std::string> why_invalid(ConfigView q) const;
  Config direction(ConfigView from, ConfigView to) const;   // short way around angular joints
  double distance(ConfigView a, ConfigView b) const;        // Euclidean norm of direction
  Config interpolate(ConfigView from, ConfigView to, double t) const;
  Config sample(Rng& rng) const override;                   // uniform; one full turn on angular joints
  std::vector<Config> unwrap_path(const std::vector<Config>& path) const;
};

}  // namespace sscbirrt
```

Contracts, identical to Python:

- Construction throws `std::invalid_argument` if `lower` and `upper` differ
  in length, any `lower[i] > upper[i]`, `angular` is nonempty with a length
  other than `dof`, or a joint **not** marked angular has a non-finite
  limit. The message for the last case is Python's: "joint i has non-finite
  limits [lo, hi]; give finite planning limits or mark it angular". Topology
  is the caller's declaration and is never inferred from the limits (#107).
- A joint is angular only if the caller says so. Its stored limits are
  ignored: it passes the limit check for any finite value, and its
  `direction` component is wrapped into $(-\pi, \pi]$ by `atan2(sin, cos)`.
  A bounded joint's component is the plain difference. Bounded joints wider
  than one turn stay bounded.
- `contains` is `why_invalid(q) == nullopt`. `why_invalid` reports, in
  order: wrong length, non-finite entry, out-of-limit joints (bounded only).
- `sample` draws each bounded joint uniformly in its limits and each angular
  joint uniformly in $[-\pi, \pi)$, whatever limits were stored for it.
  `JointSpace` is the default `SpaceSampler` of a problem.
- Uniform draws are `(rng() >> 11) * 2^-53`, not `std::uniform_real_distribution`,
  so a seeded solve is repeatable across standard libraries as well as builds.
- `unwrap_path` reproduces design.md's output rule: the first waypoint is
  returned as given, each later waypoint is the previous one plus
  `direction(prev, next)` on angular joints and the given value on bounded
  joints.

## Sets and capabilities

Membership is the only requirement. Capabilities are separate interfaces
and a set reports the ones it has through accessors that return a pointer
or `nullptr`; this is the C++ form of `supports(s, Capability)`.

```cpp
namespace sscbirrt {

struct Sample {
  Config q;
  Provenance source;
};

class SetSampler {
public:
  virtual ~SetSampler() = default;
  virtual std::vector<Sample> sample(Rng& rng) const = 0;  // candidates of one draw, unfiltered
};

class SetDistance {
public:
  virtual ~SetDistance() = default;
  virtual double distance(ConfigView q) const = 0;         // nonnegative, geometric, pre-tolerance
};

class SetViolation {
public:
  virtual ~SetViolation() = default;
  virtual double violation(ConfigView q) const = 0;        // nonnegative; zero iff contains(q)
};

class SetProjector {
public:
  virtual ~SetProjector() = default;
  virtual std::optional<Config> project(ConfigView q_previous, ConfigView q_proposed) const = 0;
};

class StateSet {
public:
  virtual ~StateSet() = default;
  virtual bool contains(ConfigView q) const = 0;

  virtual const SetSampler*   sampler()   const { return nullptr; }
  virtual const SetDistance*  distancer() const { return nullptr; }
  virtual const SetViolation* violator()  const { return nullptr; }
  virtual const SetProjector* projector() const { return nullptr; }

  // Why a capability is absent, for diagnostics. Empty if present.
  virtual std::string why_unsupported(Capability c) const;

  // Enumeration, mirroring is_finite / members / seeds.
  virtual bool is_finite() const { return false; }
  virtual std::vector<Sample> members() const { return {}; }   // exhaustive iff is_finite()
  virtual std::vector<Sample> seeds() const { return {}; }     // explicit configs anywhere inside

  virtual std::string describe() const = 0;                    // for messages; Python's repr
};

enum class Capability { Sampler, Distance, Violation, Projector };

}  // namespace sscbirrt
```

A concrete set that has a capability inherits the interface and returns
`this` from the accessor. The contracts on each method are those of
design.md's Capabilities section, restated as obligations on implementers:

- `sample` returns every candidate a draw produces and applies no external
  filter; the planner validates. It may return an empty vector.
- `distance` and `violation` are nonnegative. `violation(q) == 0.0` exactly
  when `contains(q)`.
- `project` may return `nullopt` to give up. It may use `q_previous` to seed
  or to reject a result that moved too far. The planner re-checks any
  returned configuration against the space and admissibility before use.
- All methods are `const` and must be safe to call repeatedly from one
  thread. v1.5.0 never calls a set from two threads.

Lifetime and ownership: sets are held through `std::shared_ptr<const
StateSet>`. Composites own their children the same way. The planner borrows
the problem for the duration of `solve` and stores nothing that outlives it
except the returned result, which copies configurations. A set that
captures external state (a scene, a model) is responsible for keeping that
state alive; the planner never takes ownership of it.

### Leaves

```cpp
class FiniteSet final : public StateSet, public SetSampler, public SetDistance, public SetViolation {
public:
  FiniteSet(std::vector<Config> members, double tolerance = 1e-6, Metric metric = euclidean);
  // contains: metric(q, m) <= tolerance for some member m
  // distance: min over members; violation: max(0, distance - tolerance)
  // sample: one member drawn uniformly, source {index}  (as Python's FiniteSet.sample)
  // is_finite: true; members(), seeds(): every member with source {index}
};

class EmptySet final : public StateSet {   // contains: false; is_finite: true; members: {}
};

class PredicateSet final : public StateSet {
public:
  PredicateSet(std::function<bool(ConfigView)> predicate, std::string name = "");
};
```

`FiniteSet` throws `std::invalid_argument` on an empty member list, as
Python's does ("FiniteSet requires at least one configuration"), or members
of unequal length. Members are validated against the problem's space at
`solve`, not at construction, because a set does not know its space.

### Composites

```cpp
class AnyOf final : public StateSet, public SetSampler, public SetDistance, public SetViolation, public SetProjector {
public:
  AnyOf(std::vector<std::shared_ptr<const StateSet>> children,
        std::optional<std::vector<double>> weights = std::nullopt,
        Metric metric = euclidean);
};

class IntersectionProjection {  // strategy interface
public:
  virtual ~IntersectionProjection() = default;
  virtual std::optional<Config> project(const std::vector<std::shared_ptr<const StateSet>>& children,
                                        ConfigView q_previous, ConfigView q_proposed) const = 0;
};

class IntersectionSampling {
public:
  virtual ~IntersectionSampling() = default;
  virtual std::vector<Sample> sample(const std::vector<std::shared_ptr<const StateSet>>& children, Rng& rng) const = 0;
};

class AllOf final : public StateSet, public SetSampler, public SetDistance, public SetViolation, public SetProjector {
public:
  AllOf(std::vector<std::shared_ptr<const StateSet>> children,
        std::shared_ptr<const IntersectionProjection> projection = nullptr,
        std::shared_ptr<const IntersectionSampling> sampling = nullptr);
};

class MostViolatedProjection final : public IntersectionProjection {
public:
  MostViolatedProjection(int max_iters = 50, double progress_tolerance = 1e-6);
};

class RejectionSampling final : public IntersectionSampling {
public:
  explicit RejectionSampling(int source_child);
};
```

Capabilities of a composite are computed at construction and are exactly
design.md's table. The accessors return `nullptr` for an absent capability
even though the class inherits the interface, so a caller must query the
accessor and never `dynamic_cast`. Specifically:

- One child: every accessor delegates to the child.
- `AnyOf`, several children: `distancer` and `violator` present iff every
  child has them (min over children); `sampler` present iff `weights` is
  given and every child samples (draw a child by weight, prepend its index
  to each candidate's `source`); `projector` present iff every child
  projects (project onto each, return the successful result nearest
  `q_proposed` under `metric`).
- `AllOf`, several children: `distancer` and `violator` present iff every
  child has them (max over children); `sampler` present iff a `sampling`
  strategy is given; `projector` present iff a `projection` strategy is
  given. The strategy is responsible for its own requirements: `AllOf`
  calls `projection->requires(children)` at construction, and
  `MostViolatedProjection::check_requirements` throws `UnsupportedCapability` naming
  the first child that lacks `violator` or `projector`. This is Python's
  `requires` hook at the same point.
- `is_finite`: `AnyOf` iff every child is; `AllOf` iff some child is.
  `members`: union of children's members with the child index prepended;
  intersection enumerates one finite child and keeps what the others
  contain. `seeds`: the explicit configurations anywhere in the expression,
  with full provenance, regardless of finiteness.
- Construction throws `std::invalid_argument` for an empty child list,
  weights of the wrong length, negative weights, or weights summing to zero.

`MostViolatedProjection` reproduces the Python rule exactly: repeatedly
project onto the unsatisfied child with the largest violation; progress is
the lexicographic decrease of the descending-sorted violation profile by
more than `progress_tolerance`; give up after a sweep without progress, when
a projector returns `nullopt` or leaves the point unchanged, or at
`max_iters`; succeed when every child contains the point. A satisfied child
is never selected.

### Provenance

`Provenance` semantics are Python's: `AnyOf` prepends the index of the child
it chose, `FiniteSet` contributes the index of the member, other leaves
contribute nothing. `PlanResult::start_source` and `goal_source` are the
provenance of the roots the path connects. The legacy `start_index` and
`goal_index` are the last component, or 0 if empty, computed by the binding.

## Validity and motion

```cpp
class StateValidator {
public:
  virtual ~StateValidator() = default;
  virtual bool is_valid(ConfigView q) const = 0;
};

class AcceptAll final : public StateValidator { /* true */ };

class JointBoxObstacles final : public StateValidator {
  // Invalid inside any listed axis-aligned box in joint space. The native
  // counterpart of pycbirrt.testing.Wall, so the artifact matrix runs natively.
public:
  struct Box { std::vector<double> lo, hi; };   // open on both sides, as Wall is
  explicit JointBoxObstacles(std::vector<Box> boxes);
};

struct LocalMotion {
  std::vector<Config> configs;   // validated configurations after q_from, in order
  bool reached = false;          // whole motion valid and configs.back() == q_to exactly
};

class MotionValidator {
public:
  virtual ~MotionValidator() = default;
  virtual LocalMotion validate(ConfigView q_from, ConfigView q_to) const = 0;
};

class DiscreteMotionValidator final : public MotionValidator {
public:
  DiscreteMotionValidator(const JointSpace& space, std::function<bool(ConfigView)> is_admissible, double resolution);
  // n = max(1, ceil(distance / resolution)); samples q_from + (i/n) * direction for i = 1..n;
  // sample n is q_to exactly; stops at the first inadmissible sample with reached = false.
};

class RestrictedMotionValidator final : public MotionValidator {
public:
  RestrictedMotionValidator(std::shared_ptr<const MotionValidator> base,
                            std::function<bool(ConfigView, ConfigView)> accepts);
};
```

`LocalMotion` carries Python's contract: with `reached == true` on a nonzero
motion, `configs` is nonempty and `configs.back()` equals `q_to` exactly
(bitwise, as `np.array_equal`). The planner checks this before touching a
tree and throws `ContractError` on violation. A custom validator replaces
the default and owns the interior of the motion; the planner still checks
every returned configuration for admissibility, rejects a `reached` motion
whole if any is inadmissible, and keeps the admissible prefix of an
unreached one.

## The problem

```cpp
struct PlanningProblem {
  std::shared_ptr<const JointSpace>      space;
  std::shared_ptr<const StateSet>        start;
  std::shared_ptr<const StateSet>        goal;
  std::shared_ptr<const StateValidator>  validator;
  std::shared_ptr<const StateSet>        path_constraint;   // may be null
  std::shared_ptr<const MotionValidator> motion_validator;  // null means the default
  std::shared_ptr<const SpaceSampler>    sampler;           // null means *space
};
```

Roles are Python's, and so are the two replaceable strategy components:
the motion validator and the free-space sampler (#110). The sampler
proposes the targets the trees grow toward; start and goal bias remain the
planner's and mix the role sets' own samplers with it, and the sampler is
not consulted for roots or bias draws. A target outside the space is
handled by not growing toward it. Replacing the default trades away
probabilistic completeness unless the replacement has full support over
the space; that is the caller's responsibility. The problem is a value type; copying it shares the
components.

## Planner configuration

```cpp
struct PlannerConfig {
  // Termination
  double timeout_seconds = 30.0;
  int    max_iterations  = 100000;

  // Tolerances
  double connection_tolerance          = 1e-3;
  std::optional<double> edge_resolution;      // nullopt means step_size
  double progress_tolerance            = 1e-6;

  // Growth
  double step_size  = 0.1;
  double goal_bias  = 0.1;
  double start_bias = 0.1;
  std::optional<int> extend_steps;            // nullopt: connect until blocked
  std::optional<int> connect_steps;

  // Roots
  int sample_draws     = 100;   // Python: tsr_samples, the draw budget per role
  int num_tree_roots   = 100;
  int max_per_draw     = 3;     // Python: max_ik_per_pose

  // Smoothing
  bool smooth_path         = true;
  int  smoothing_iterations = 50;
  int  smoothing_patience   = 15;
};
```

Three Python fields are absent on purpose. `membership_tolerance`,
`max_projection_iters`, and `projection_progress_tolerance` belong to the
concrete sets and strategies (design.md: "membership tolerance belongs to
the concrete set"); the legacy lowering copies them into the sets it builds,
and the binding does the same. `angular_joints` lives on `JointSpace` only.
`abort_fn` is replaced by a cancellation token. `tsr_tolerance` is a
deprecated alias with no native form.

Ingress validation of the config, at `solve`, throws `std::invalid_argument`
with Python's message shape ("<field> must be <requirement>, got <value>")
and Python's ranges (#108): positive `timeout_seconds`, `step_size`,
`progress_tolerance`; nonnegative `connection_tolerance`,
`smoothing_iterations`, `smoothing_patience`; at least 1 for
`max_iterations`, `sample_draws`, `num_tree_roots`, `max_per_draw`;
`edge_resolution` empty or positive; `extend_steps` and `connect_steps`
empty or at least 1; `goal_bias` and `start_bias` within $[0, 1]$. The
ranges Python checks on its set-owned fields (`membership_tolerance`
nonnegative, `max_projection_iters` at least 1,
`projection_progress_tolerance` positive) are enforced natively by the
constructors that own those values, `FiniteSet`, the TSR set in v1.6.0, and
`MostViolatedProjection`.

## Deadlines, cancellation, randomness

```cpp
class CancellationToken {
public:
  void cancel() noexcept;            // any thread
  bool cancelled() const noexcept;   // std::atomic<bool>, relaxed
};

struct SolveOptions {
  std::optional<std::uint64_t> seed;                    // nullopt: nondeterministic seed
  std::shared_ptr<const CancellationToken> cancel;      // may be null
};
```

- The solver owns one `std::mt19937_64` per `solve`, seeded from `seed`.
  Every random decision in the solve draws from it, in a fixed order, and
  every set receives it by reference. With the same seed, the same problem,
  and the same build, a single-threaded solve is repeatable. Python uses
  numpy's PCG64, so a native path is never expected to equal a Python path;
  the artifact compares semantics (#92), not waypoints.
- The deadline is `std::chrono::steady_clock::now() + timeout_seconds`,
  taken when the search loop starts (after roots, as Python does). It is
  checked once per iteration.
- The token is checked at Python's three points (#109): once per search
  iteration before the deadline, before each sampling draw during root
  collection, and before each smoothing attempt. The outcomes are Python's:
  during roots, `Status::Aborted` with zero iterations, `reason` "Aborted by
  user during <role> root collection", and trees holding the roots gathered
  so far; during the search, `Status::Aborted`; during smoothing, smoothing
  stops and the result is `Status::Success` with the path as smoothed so
  far, because a valid path exists. Finite sets involve no draws and are
  not polled during roots. Cancellation is cooperative: a set or validator
  that runs long is not interrupted.
- The solver does not spawn threads and calls no set from more than one
  thread.

## Result and statuses

```cpp
enum class Status {
  Success,
  Timeout,
  Aborted,
  MaxIterations,
};

struct RootReport {                 // per role; carried by the result and by NoRoots
  int explicit_candidates = 0;      // from seeds()
  int explicit_rejected   = 0;
  int draws               = 0;
  int draws_empty         = 0;      // "IK unreachable" in Python's summary
  int outside_space       = 0;
  int in_collision        = 0;
  int constraint_violated = 0;
  int roots               = 0;
  bool only_collisions() const;     // every rejection was the validator's
  std::vector<std::string> details; // per explicit rejection, as Python logs them
};

struct PlanResult {
  Status status;
  std::string reason;                  // human-readable; empty on success
  std::vector<Config> path;            // empty unless Success; unwrapped per JointSpace rule
  Provenance start_source, goal_source;
  int iterations = 0;
  double planning_seconds = 0.0;
  std::pair<int, int> tree_sizes{0, 0};
  RootReport start_roots, goal_roots;
  std::shared_ptr<const Tree> tree_start, tree_goal;   // for inspection; may be null if not requested
};
```

Ordinary search outcomes are statuses. Exceptions are reserved for:

| Exception | When |
|---|---|
| `std::invalid_argument` | malformed input at ingress: dimensions, non-finite values, bounds, tolerances, option combinations, a set whose members do not match the space's `dof` |
| `sscbirrt::UnsupportedCapability` | a start or goal set that is neither finite nor sampleable; a composite asked for a capability it lacks; a strategy whose children lack what it needs |
| `sscbirrt::NoRoots` | no admissible root for a role; carries the role and its `RootReport` |
| `sscbirrt::ContractError` | a `MotionValidator` violated the `LocalMotion` contract; a set or projector returned a configuration of the wrong length |

No admissible root is an exception in both backends, as Python has it:
whether a role has any root is a property of the problem the planner
discovers before the search starts, and callers distinguish it from a
search that ran and failed. The binding maps `NoRoots` to Python's four
exceptions: `AllStartConfigurationsInCollision` when
`report.only_collisions()`, otherwise `AllStartConfigurationsInvalid`, and
likewise for the goal, with the report's details as the message. A role
with no explicit members and nothing to draw from (an `EmptySet`) is
`std::invalid_argument`, Python's `ValueError` "No valid <role>
configurations available". Whether these become statuses is a 2.0 decision
to be made for both backends at once.

`reason` strings begin with the same prefixes the artifact's
`failure_category` recognizes: `Timeout`, `Aborted`, `Max iterations`.

## The search, as obligations

The native search is Python's, function for function. This list is the
contract; `planner.py` is the reference for anything it leaves open.

1. **Ingress.** Validate the config. For each of start and goal, require
   `is_finite() || sampler() != nullptr`, else throw `UnsupportedCapability`.
   Check every explicit member (`seeds`) has length `dof`.
2. **Admissibility** of a configuration is, in order: `space.contains`,
   `validator.is_valid`, `path_constraint.contains` (if any). The order is
   observable through `RootReport` and must be kept.
3. **Roots** for a role: every `seeds()` candidate that is admissible, in
   order, with its provenance. Then, if the set is not finite and samples,
   draw up to `sample_draws` times or until `num_tree_roots` roots exist,
   keeping at most `max_per_draw` admissible candidates per draw, skipping
   a candidate whose provenance equals an explicit seed's. Rejections are
   counted in the `RootReport`. No roots for a role throws `NoRoots`.
   Before each draw the cancellation token is checked; if set, the solve
   returns `Status::Aborted` with the roots gathered so far.
4. **Iteration** `i` extends the start tree if `i` is even, else the goal
   tree. Cancellation is checked, then the deadline. The target is a sample
   from the opposite role's set with probability `goal_bias` or
   `start_bias` (only if that set samples; up to `sample_draws` attempts to
   find an admissible one), otherwise one draw from `problem.sampler`, or
   from `space` when the sampler is null.
5. **Growth** toward a target from the nearest node under `space.distance`
   (ties to the lowest index): if the target is outside the space, no
   growth. Otherwise repeat: if the remaining distance is exactly zero,
   reached without adding a node; if it is below `connection_tolerance`,
   extend along the exact final edge to the target and return its outcome;
   if the distance shrank by less than `progress_tolerance`, stop; if the
   step budget (`extend_steps` or `connect_steps`) is spent, stop; take a
   step of at most `step_size` along `direction`; if the step leaves the
   space, stop; if the constraint has a projector, project from the current
   node and stop on `nullopt` or a result outside the space; extend along
   the edge to the new configuration and stop if not reached.
6. **Edges** are the single local-motion boundary: growth, the final
   connection, and shortcuts all go through `motion_validator.validate`
   (default: `DiscreteMotionValidator` at `edge_resolution` or `step_size`
   over the admissibility predicate). Zero-length motions succeed without a
   node. Returned configurations are stored as consecutive nodes after the
   contract and admissibility checks above.
7. **Connection.** After extending tree A to a node, grow tree B toward that
   node's configuration with the connect budget. Connected means tree B now
   holds that exact configuration through a validated edge.
8. **Extraction** joins the start-tree path to the meeting node with the
   reversed goal-tree path, dropping the duplicated join configuration.
9. **Smoothing**, if enabled: up to `smoothing_iterations` attempts, each
   choosing `i` uniformly in `[0, n-3]` and `j` uniformly in `[i+2, n-1]`,
   growing a fresh single-root tree from `path[i]` toward `path[j]` with no
   step budget; replace the segment only if the shortcut's path length under
   `space.distance` is smaller by more than `1e-9`. Stop after
   `smoothing_patience` attempts without improvement, when the path has two
   waypoints, or when the cancellation token is set before an attempt; in
   the last case the result is still `Success` with the path so far.
10. **Output.** `space.unwrap_path` on the final path. `start_source` and
    `goal_source` are the provenance of the roots the meeting node descends
    from in each tree.

The `Tree` type is `struct Node { Config q; int parent; Provenance source; }`
in a vector, with `nearest` a linear scan. A spatial index is permitted
later only if it returns the same node as the linear scan, ties included.

## Python binding

The extension module is `pycbirrt._native`, built with pybind11 as ssik's
is. It exposes the types above under the same names and one function:

```python
pycbirrt._native.Planner(config: _native.PlannerConfig).solve(
    problem: _native.PlanningProblem, seed: int | None, cancel: _native.CancellationToken | None,
    keep_trees: bool = True) -> _native.PlanResult
```

The GIL is released for the duration of `solve`. After entry, the native
solve touches no Python object; that is why every component must be a
native object. The public Python surface in v1.5.0 is:

```python
from pycbirrt.backends.native import lower, NativeUnsupported

lowered = lower(problem, config)       # PlanningProblem + CBiRRTConfig -> native problem, or raises NativeUnsupported
planner = CBiRRT(robot, ik, collision, config, backend="native")   # "python" (default) | "native" | "auto"
```

A note for the parity gate (#92): provenance is a property of the draws
when a role has more than one admissible root, so the two backends compare
it only where the reached root is unique; otherwise the check is that the
reported provenance names one of the admissible roots and matches the
path's endpoint. The artifact's `nested_finite_goal` case is the example:
Python's seed reaches the near member, a native seed may reach the far one,
and both are correct.

`lower` walks the Python problem and maps each component:

| Python | Native | Otherwise |
|---|---|---|
| `JointSpace` | `JointSpace` | |
| `FiniteSet`, `EmptySet` | same | |
| `AnyOf`, `AllOf` with `MostViolatedProjection` / `RejectionSampling` | same, children lowered recursively | |
| `PredicateSet`, `TSRConfigurationSet`, any other set | | `NativeUnsupported("goal: TSRConfigurationSet has no native form in v1.5.0")` |
| validator: `pycbirrt.testing.NoCollision`, `Wall` | `AcceptAll`, `JointBoxObstacles` | any other validator: `NativeUnsupported("validator: <type> is a Python object; native needs a sscbirrt.StateValidator")` |
| `motion_validator` None | default | any custom validator: `NativeUnsupported` |
| `sampler` None | default (`space`) | any custom sampler: `NativeUnsupported("sampler: <type> is a Python object")` |
| `CBiRRTConfig` | `PlannerConfig` (field map above), `abort_fn` wrapped in a token polled from a Python thread | |

`NativeUnsupported` carries the list of every component that blocked
lowering, not only the first. With `backend="auto"`, the planner catches it,
selects the Python backend, records the reasons on
`PlanResult.backend_reasons`, and sets `PlanResult.backend = "python"`; with
`backend="native"` it propagates; with the default `"python"` no lowering is
attempted. Default `plan(...)` behavior is therefore unchanged in 1.x. v2.0.0
flips the default to `"auto"` under #86.

Native results are converted to Python `PlanResult`s: `Status::Success` and
the three search failures map to `success` and `failure_reason` with the
same reason prefixes; `NoRoots` becomes the Python exception described
above; `std::invalid_argument` becomes `ValueError`, `UnsupportedCapability`
and `ContractError` their Python namesakes; trees are wrapped read-only. `pycbirrt` imports and works without
the extension present; `backend="native"` then raises `NativeUnsupported`
naming the missing module.

## CMake targets and packaging

```
cpp/
  CMakeLists.txt                 project(sscbirrt LANGUAGES CXX), C++20
  include/sscbirrt/*.hpp         public headers
  src/*.cpp                      core implementation
  bindings/pycbirrt_native.cpp   pybind11 module (only target that sees Python)
  tests/*.cpp                    ctest targets
  examples/consumer/             standalone find_package consumer, built against the installed package
```

| Target | Kind | Depends on | Installed |
|---|---|---|---|
| `sscbirrt::core` | static library (`BUILD_SHARED_LIBS` respected) | C++20 standard library | yes, with headers and `sscbirrtConfig.cmake` |
| `sscbirrt_tests` | executables under ctest | `sscbirrt::core` | no |
| `sscbirrt_consumer` | executable, `examples/consumer/` | `find_package(sscbirrt)` | no |
| `pycbirrt_native` | pybind11 module `pycbirrt._native` | `sscbirrt::core`, pybind11, Python | into the wheel |

Options: `SSCBIRRT_BUILD_TESTS` (default ON in-tree), `SSCBIRRT_BUILD_PYTHON`
(default OFF; the wheel build turns it on), `SSCBIRRT_SANITIZE` (adds
`-fsanitize=address,undefined` to tests; CI runs the test target with it,
as #85 requires). Warnings are `-Wall -Wextra -Wpedantic -Werror` on the
core and tests.

The wheel is built by scikit-build-core driving this CMake project with
`SSCBIRRT_BUILD_PYTHON=ON`, replacing the current setuptools backend. ssik
builds its extension from a hatchling hook instead; the difference here is
that the C++ package must be installable and consumable on its own (#85's
consumer criterion), so CMake is the primary build and the wheel reuses it
rather than the other way around. This is a decision for review.

The consumer example is the acceptance test of the export: it is built
against `cmake --install`'s output, not the source tree, exactly as ssik's
`cpp/examples/consumer` is.

## Minimal standalone consumer

```cpp
#include <sscbirrt/sscbirrt.hpp>
#include <cstdio>

int main() {
  using namespace sscbirrt;
  auto space = std::make_shared<JointSpace>(std::vector<double>{-3.14159, -3.14159},
                                            std::vector<double>{ 3.14159,  3.14159});
  auto start = std::make_shared<FiniteSet>(std::vector<Config>{{-2.0, 0.5}});
  auto goal  = std::make_shared<FiniteSet>(std::vector<Config>{{2.0, -0.5}, {2.0, 0.5}});
  auto walls = std::make_shared<JointBoxObstacles>(std::vector<JointBoxObstacles::Box>{
      {{-0.2, -3.2}, {0.2, 1.0}}});   // a slab at q0 in (-0.2, 0.2) for q1 below 1.0

  PlanningProblem problem{space, start, goal, walls, nullptr, nullptr};
  PlannerConfig config;
  config.step_size = 0.2;
  config.timeout_seconds = 5.0;

  Planner planner(config);
  PlanResult r = planner.solve(problem, SolveOptions{.seed = 0});
  if (r.status != Status::Success) {
    std::printf("no path: %s\n", r.reason.c_str());
    return 1;
  }
  std::printf("%zu waypoints; goal member %d\n", r.path.size(), r.goal_source.back());
  return 0;
}
```

```cmake
cmake_minimum_required(VERSION 3.16)
project(consumer LANGUAGES CXX)
find_package(sscbirrt REQUIRED)
add_executable(plan main.cpp)
target_link_libraries(plan PRIVATE sscbirrt::core)
```

## Concept map

Every native concept maps to a Python concept. "same" means the contract is
identical; otherwise the difference and its reason are stated.

| Python | Native | Relation |
|---|---|---|
| `JointSpace(lower, upper, angular_joints)` | `JointSpace` | same: finite limits required on bounded joints, angular joints ignore theirs and sample one full turn (#107) |
| `CBiRRTConfig.angular_joints` | on `JointSpace` only | same: in Python too the space owns the topology and the config field only feeds the legacy constructor; the binding does what `CBiRRT.__init__` does |
| `PlanningProblem.sampler`, `SpaceSampler` | same | same (#110) |
| `StateSet.contains` | `StateSet::contains` | same |
| `supports(s, Cap)` | `s.sampler() != nullptr` etc. | same meaning; accessor instead of protocol check |
| `Sample(q, source)` | `Sample{q, source}` | same |
| `SetSampler/Distance/Violation/Projector` | same names | same contracts |
| `FiniteSet(configs, tolerance, metric)` | `FiniteSet` | same, including rejecting an empty member list at construction |
| `PredicateSet(fn)` | `PredicateSet(std::function)` | same in C++; not lowerable from Python in v1.5.0 (would need a callback) |
| `EmptySet` | `EmptySet` | same |
| `AnyOf`, `AllOf`, strategies | same | same rules; strategy requirements checked at `AllOf` construction through `requires`, as in Python |
| `is_finite`, `members`, `seeds` | virtual methods | same |
| `CollisionChecker.is_valid` | `StateValidator::is_valid` | same; renamed because validity is not only collision |
| `pycbirrt.testing.NoCollision`, `Wall` | `AcceptAll`, `JointBoxObstacles` | same predicates; `Wall`'s `extent` becomes a second box dimension |
| `LocalMotion`, `MotionValidator`, `DiscreteMotionValidator`, `RestrictedMotionValidator` | same | same contracts |
| `PlanningProblem` | `PlanningProblem` | same roles; `shared_ptr<const>` ownership |
| `CBiRRTConfig` | `PlannerConfig` | same ranges and messages (#108); two fields renamed where the Python name was TSR-specific (`tsr_samples` → `sample_draws`, `max_ik_per_pose` → `max_per_draw`); set-owned tolerances live on the sets, which is where Python's lowering puts them |
| `abort_fn` | `CancellationToken` | same three polling points and outcomes (#109) |
| `seed` | `SolveOptions::seed`, `std::mt19937_64` | **the one difference**: Python uses numpy's PCG64. Same seed gives a repeatable solve within each backend, never the same path across them; the artifact compares semantics. A PCG64 port would remove this at the cost of coupling every set's draw order to numpy internals, and #85 lists waypoint equality as a non-goal. |
| `PlanResult.success/failure_reason` | `Status` + `reason` | same information; `reason` prefixes preserved |
| `AllStart/Goal...Invalid/InCollision`, `ValueError` for an empty role | `NoRoots` + `RootReport`, `std::invalid_argument` | same: exceptions in both; the binding maps them by name and by `only_collisions()` |
| `UnsupportedCapability`, `MotionContractError` | `UnsupportedCapability`, `ContractError` | same triggers |
| `RRTree`, `Node` | `Tree`, `Node` | same; nearest with lowest-index tie-break |
| `unwrap_path` output rule | same | same |

## Open questions for review

1. **Core without Eigen.** Plain `std::vector<double>` keeps the consumer
   story dependency-free. If v1.6.0 pose regions want Eigen in the core
   rather than in `sscbirrt::tsr`, that is a later, separate decision.
2. **scikit-build-core versus a hatchling hook** for the wheel. Argued above;
   the alternative keeps pycbirrt on one build tool with ssik.
3. **Exposing trees.** Kept because Python exposes them and the artifact
   tooling inspects them. They cost a copy per solve; a flag on
   `SolveOptions` can suppress it.
4. **`PredicateSet` in the binding.** Excluded in v1.5.0 to honor the
   no-callback rule. A later release could allow it with the GIL held and a
   documented cost.
