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


@pytest.mark.parametrize("script", ["planar_arm.py", "multi_config_demo.py"])
def test_example_runs_headless(script, tmp_path):
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        cwd=tmp_path,  # examples write images to the working directory
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
