# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Smoke-run the examples that need no simulator, so API drift cannot break them silently (#32)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_planar_example_runs_headless(tmp_path):
    """All four scenarios, including several starts and goals (formerly multi_config_demo.py, #166)."""
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "planar_arm.py")],
        cwd=tmp_path,  # examples write images to the working directory
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    for example in ("Example 1", "Example 2", "Example 3", "Example 4"):
        assert example in proc.stdout
    assert "Connected start" in proc.stdout
