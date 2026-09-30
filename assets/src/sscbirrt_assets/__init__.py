# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Robot models for the sscbirrt demos: the MuJoCo Menagerie UR5e and Robotiq 2F-85.

The files are copied unmodified from google-deepmind/mujoco_menagerie at the commit in
:data:`MENAGERIE_COMMIT`, each under its own LICENSE (UR5e: BSD-3-Clause; 2F-85: BSD-2-Clause).
The functions return real file paths because loaders such as ``ssik.Manipulator.from_mjcf`` and
``mujoco.MjSpec.from_file`` resolve the meshes relative to the MJCF.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["MENAGERIE_COMMIT", "menagerie_path", "robotiq_2f85_xml", "ur5e_xml"]

MENAGERIE_COMMIT = "feadf76d42f8a2162426f7d226a3b539556b3bf5"

_ROOT = Path(__file__).resolve().parent / "menagerie"


def menagerie_path() -> Path:
    """A directory laid out like a menagerie clone, holding only the shipped models."""
    return _ROOT


def _existing(relative: str) -> Path:
    path = _ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. In a source checkout, run `python assets/fetch.py` to place the models."
        )
    return path


def ur5e_xml() -> Path:
    """The Menagerie UR5e MJCF (``universal_robots_ur5e/ur5e.xml``)."""
    return _existing("universal_robots_ur5e/ur5e.xml")


def robotiq_2f85_xml() -> Path:
    """The Menagerie Robotiq 2F-85 MJCF (``robotiq_2f85/2f85.xml``)."""
    return _existing("robotiq_2f85/2f85.xml")
