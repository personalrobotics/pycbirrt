# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Property-based tests for state-set laws over random nested compositions.

The example tests in test_sets.py document intended behavior at chosen
points. These tests sweep random trees of AnyOf/AllOf over random 1-D
leaves and random configurations, checking the laws that define the sets.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from pycbirrt.sets import (
    AllOf,
    AnyOf,
    FiniteSet,
    MostViolatedProjection,
    PredicateSet,
    RejectionSampling,
    Sample,
    SetDistance,
    SetProjector,
    SetSampler,
    SetViolation,
    supports,
)

TOL = 1e-9


# ---------------------------------------------------------------------------
# 1-D leaf sets
# ---------------------------------------------------------------------------


class Interval:
    """{q : lo <= q[0] <= hi}: distance and projection, no sampler."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def distance(self, q):
        x = float(q[0])
        return max(0.0, self.lo - x, x - self.hi)

    def violation(self, q):
        return max(0.0, self.distance(q) - TOL)

    def contains(self, q):
        return self.distance(q) <= TOL

    def project(self, q_previous, q_proposed):
        return np.array([float(np.clip(q_proposed[0], self.lo, self.hi))])

    def __repr__(self):
        return f"Interval({self.lo}, {self.hi})"


class Halfspace(Interval):
    """{q : q[0] >= lo}: distance, projection, and sampling from [lo, lo + 1]."""

    def __init__(self, lo):
        super().__init__(lo, float("inf"))

    def sample(self, rng):
        return [Sample(np.array([rng.uniform(self.lo, self.lo + 1.0)]))]

    def __repr__(self):
        return f"Halfspace({self.lo})"


finite_floats = st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False)


@st.composite
def intervals(draw):
    a, b = sorted([draw(finite_floats), draw(finite_floats)])
    return Interval(a, b)


halfspaces = st.builds(Halfspace, finite_floats)

finite_sets = st.lists(finite_floats, min_size=1, max_size=3).map(
    lambda xs: FiniteSet([np.array([x]) for x in xs], tolerance=TOL)
)

predicates = finite_floats.map(lambda c: PredicateSet(lambda q, c=c: q[0] > c, name=f"q>{c}"))

leaves = st.one_of(intervals(), halfspaces, finite_sets, predicates)


@st.composite
def any_ofs(draw, children):
    kids = draw(st.lists(children, min_size=1, max_size=3))
    weights = None
    if len(kids) > 1 and draw(st.booleans()):
        n = len(kids)
        weights = draw(st.lists(st.floats(0.0, 1.0), min_size=n, max_size=n).filter(lambda w: sum(w) > 0))
    return AnyOf(kids, weights=weights)


@st.composite
def all_ofs(draw, children):
    kids = draw(st.lists(children, min_size=1, max_size=3))
    projection = None
    projectable = all(supports(k, SetViolation) and supports(k, SetProjector) for k in kids)
    if len(kids) > 1 and projectable and draw(st.booleans()):
        projection = MostViolatedProjection()
    sampling = None
    sources = [i for i, k in enumerate(kids) if supports(k, SetSampler)]
    if len(kids) > 1 and sources and draw(st.booleans()):
        sampling = RejectionSampling(source=draw(st.sampled_from(sources)))
    return AllOf(kids, projection=projection, sampling=sampling)


trees = st.recursive(leaves, lambda ch: st.one_of(any_ofs(ch), all_ofs(ch)), max_leaves=8)

configs = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False).map(lambda x: np.array([x]))


# ---------------------------------------------------------------------------
# Reference semantics, computed from the tree structure
# ---------------------------------------------------------------------------


def ref_contains(node, q):
    if isinstance(node, AnyOf):
        return any(ref_contains(c, q) for c in node.children)
    if isinstance(node, AllOf):
        return all(ref_contains(c, q) for c in node.children)
    return node.contains(q)


def flatten(node):
    """Flatten nested same-operator composites into one level (an equivalent set)."""
    if isinstance(node, (AnyOf, AllOf)):
        kind = type(node)
        kids = []
        for c in node.children:
            c = flatten(c)
            if isinstance(c, kind):
                kids.extend(c.children)
            else:
                kids.append(c)
        return kind(kids)
    return node


def walk_provenance(node, source, q):
    """Follow a sample's source path down the tree; the leaf must contain q."""
    if isinstance(node, AnyOf):
        i, *rest = source
        walk_provenance(node.children[i], tuple(rest), q)
    elif isinstance(node, AllOf):
        child = node.children[0] if len(node.children) == 1 else node.children[node.sampling.source]
        walk_provenance(child, source, q)
    elif isinstance(node, FiniteSet):
        (i,) = source
        assert np.array_equal(q, node.configs[i])
    else:
        assert source == ()
        assert node.contains(q)


# ---------------------------------------------------------------------------
# Laws
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(trees, configs)
def test_membership_matches_boolean_semantics(tree, q):
    assert tree.contains(q) == ref_contains(tree, q)


@settings(max_examples=300, deadline=None)
@given(trees, configs)
def test_flattening_same_operator_preserves_membership(tree, q):
    assert flatten(tree).contains(q) == tree.contains(q)


@settings(max_examples=300, deadline=None)
@given(trees, configs)
def test_distance_agrees_with_membership(tree, q):
    if not supports(tree, SetDistance):
        return
    d = tree.distance(q)
    assert d >= 0.0
    assert tree.contains(q) == (d <= TOL)


@settings(max_examples=300, deadline=None)
@given(trees, configs)
def test_violation_is_zero_iff_contains(tree, q):
    if not supports(tree, SetViolation):
        return
    v = tree.violation(q)
    assert v >= 0.0
    assert tree.contains(q) == (v == 0.0)


@settings(max_examples=300, deadline=None)
@given(trees, configs)
def test_composite_violation_bounds_children(tree, q):
    if not isinstance(tree, (AnyOf, AllOf)) or not supports(tree, SetViolation):
        return
    v = tree.violation(q)
    child_vs = [c.violation(q) for c in tree.children]
    if isinstance(tree, AnyOf):
        assert all(v <= cv for cv in child_vs)
    else:
        assert all(v >= cv for cv in child_vs)


@settings(max_examples=300, deadline=None)
@given(trees, configs)
def test_composite_distance_bounds_children(tree, q):
    if not isinstance(tree, (AnyOf, AllOf)) or not supports(tree, SetDistance):
        return
    d = tree.distance(q)
    child_ds = [c.distance(q) for c in tree.children]
    if isinstance(tree, AnyOf):
        assert all(d <= cd for cd in child_ds)
    else:
        assert all(d >= cd for cd in child_ds)


@settings(max_examples=300, deadline=None)
@given(trees, configs, configs)
def test_projection_lands_in_set(tree, q_prev, q):
    if not supports(tree, SetProjector):
        return
    result = tree.project(q_prev, q)
    if result is not None:
        assert tree.contains(result)


@settings(max_examples=300, deadline=None)
@given(trees, st.integers(0, 2**32 - 1))
def test_samples_are_members_with_valid_provenance(tree, seed):
    if not supports(tree, SetSampler):
        return
    candidates = tree.sample(np.random.default_rng(seed))
    # Leaf samplers always succeed; rejection sampling inside an AllOf may legitimately reject all
    for smp in candidates:
        assert tree.contains(smp.q)
        walk_provenance(tree, smp.source, smp.q)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(halfspaces, min_size=2, max_size=4),
    st.data(),
    st.integers(0, 2**32 - 1),
)
def test_zero_weight_children_are_never_chosen(kids, data, seed):
    weights = data.draw(
        st.lists(st.sampled_from([0.0, 1.0]), min_size=len(kids), max_size=len(kids)).filter(lambda w: sum(w) > 0)
    )
    s = AnyOf(kids, weights=weights)
    rng = np.random.default_rng(seed)
    for _ in range(20):
        i = s.sample(rng)[0].source[0]
        assert weights[i] > 0


@settings(max_examples=300, deadline=None)
@given(trees)
def test_unsupported_capabilities_are_reported_consistently(tree):
    """supports() is False exactly when a composite would refuse the call."""
    if not isinstance(tree, (AnyOf, AllOf)):
        return
    single = len(tree.children) == 1
    if single:
        for cap in (SetSampler, SetDistance, SetProjector):
            assert supports(tree, cap) == supports(tree.children[0], cap)
    else:
        all_dist = all(supports(c, SetDistance) for c in tree.children)
        assert supports(tree, SetDistance) == all_dist
        assert supports(tree, SetViolation) == all(supports(c, SetViolation) for c in tree.children)
        if isinstance(tree, AnyOf):
            all_proj = all(supports(c, SetProjector) for c in tree.children)
            all_samp = all(supports(c, SetSampler) for c in tree.children)
            assert supports(tree, SetProjector) == all_proj
            assert supports(tree, SetSampler) == (all_samp and tree.weights is not None)
        else:
            assert supports(tree, SetSampler) == (tree.sampling is not None)
            assert supports(tree, SetProjector) == (tree.projection is not None)
