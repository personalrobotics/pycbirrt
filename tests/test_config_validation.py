# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""CBiRRTConfig rejects malformed ranges at construction (#108)."""

import pytest

from sscbirrt import CBiRRTConfig


def test_defaults_construct():
    CBiRRTConfig()


@pytest.mark.parametrize(
    "field, value, requirement",
    [
        ("timeout", 0.0, "positive"),
        ("timeout", -1.0, "positive"),
        ("step_size", 0.0, "positive"),
        ("progress_tolerance", 0.0, "positive"),
        ("projection_progress_tolerance", 0.0, "positive"),
        ("membership_tolerance", -1e-9, "nonnegative"),
        ("connection_tolerance", -1e-9, "nonnegative"),
        ("max_iterations", 0, "at least 1"),
        ("tsr_samples", 0, "at least 1"),
        ("num_tree_roots", 0, "at least 1"),
        ("max_ik_per_pose", 0, "at least 1"),
        ("max_projection_iters", 0, "at least 1"),
        ("smoothing_iterations", -1, "nonnegative"),
        ("smoothing_patience", -1, "nonnegative"),
        ("edge_resolution", 0.0, "None or positive"),
        ("extend_steps", 0, "None or at least 1"),
        ("connect_steps", 0, "None or at least 1"),
        ("goal_bias", 1.5, r"within \[0, 1\]"),
        ("start_bias", -0.1, r"within \[0, 1\]"),
    ],
)
def test_out_of_range_field_is_rejected_by_name(field, value, requirement):
    with pytest.raises(ValueError, match=rf"^{field} must be {requirement}, got"):
        CBiRRTConfig(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("membership_tolerance", 0.0),
        ("connection_tolerance", 0.0),
        ("smoothing_iterations", 0),
        ("smoothing_patience", 0),
        ("goal_bias", 0.0),
        ("start_bias", 1.0),
        ("edge_resolution", None),
        ("extend_steps", 1),
        ("connect_steps", None),
    ],
)
def test_boundary_values_are_accepted(field, value):
    CBiRRTConfig(**{field: value})


def test_deprecated_alias_is_validated_too():
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="membership_tolerance must be nonnegative"):
            CBiRRTConfig(tsr_tolerance=-1.0)
