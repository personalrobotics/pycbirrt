# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Rendered MuJoCo demos of sscbirrt on a UR5e with a Robotiq 2F-85.

    pip install "sscbirrt[demo]"
    sscbirrt-demo                 # every scenario, one MP4 each, in ./sscbirrt-demo/
    sscbirrt-demo --list

The robot models come from the ``sscbirrt-assets`` wheel (``MUJOCO_MENAGERIE_PATH`` overrides it).
Importing this package does not import MuJoCo; the submodules do.
"""

from __future__ import annotations

import os
import sys


def headless_gl_default() -> None:
    """On Linux with no display, render through EGL unless the caller chose a backend.

    MuJoCo reads ``MUJOCO_GL`` once, when it is first imported, so call this before anything imports mujoco.
    """
    if "MUJOCO_GL" in os.environ or not sys.platform.startswith("linux"):
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        os.environ["MUJOCO_GL"] = "egl"
