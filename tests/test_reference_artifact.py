# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The Python reference artifact pins the planner's semantics on a fixed matrix (#94).

Regenerates the artifact in-process and compares its semantic view with the
checked-in file: status, failure category, provenance, and the independent
validation report. Paths and iteration counts are inspectable but not
compared, so the test is stable across platforms while behavior is pinned.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "reference_artifact.py"
ARTIFACT = ROOT / "tests" / "reference" / "python_reference.json"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("reference_artifact", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["reference_artifact"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fresh(tool):
    return tool.generate()


@pytest.fixture(scope="module")
def stored():
    return json.loads(ARTIFACT.read_text())


def test_artifact_is_checked_in(stored):
    assert stored["cases"], "artifact has no cases"
    assert {"pycbirrt", "sstsr", "numpy", "python"} <= set(stored["versions"])


def test_semantics_match_the_stored_artifact(tool, fresh, stored):
    assert tool.semantic_view(fresh) == tool.semantic_view(stored)


def test_every_success_passes_independent_validation(fresh):
    for case in fresh["cases"]:
        if case["status"] != "success":
            continue
        v = case["validation"]
        for key in (
            "all_in_space",
            "first_in_start_set",
            "last_in_goal_set",
            "all_admissible",
            "edges_validated_at_resolution",
            "raw_steps_within_step_size",
        ):
            assert v[key], f"{case['name']}: {key} failed"


def test_expected_statuses(fresh):
    by_name = {c["name"]: c for c in fresh["cases"]}
    assert by_name["timeout"]["failure_category"] == "timeout"
    assert by_name["cancellation"]["failure_category"] == "aborted"
    assert by_name["unreachable"]["failure_category"] == "max_iterations"
    for name in (
        "fixed_to_fixed",
        "multiple_roots",
        "nested_finite_goal",
        "wrapped_seam",
        "rejection_constraint",
        "projected_constraint",
        "tsr_goal_union",
        "tsr_chain_goal",
        "allof_constraint",
    ):
        assert by_name[name]["status"] == "success", name
    assert by_name["nested_finite_goal"]["goal_source"] == [0, 1]  # the filtered member of the first alternative


def test_regeneration_is_deterministic(tool):
    a, b = tool.generate(), tool.generate()
    a.pop("versions"), b.pop("versions")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_check_mode_passes(tool, capsys):
    assert tool.main(["--check"]) == 0
    assert "semantics match" in capsys.readouterr().out
