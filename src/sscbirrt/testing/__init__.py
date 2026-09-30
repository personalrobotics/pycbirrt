# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Small deterministic robots for tests, examples, and the reference artifact."""

from sscbirrt.testing.planar import NoCollision, PlanarArm, PlanarIK, Wall

__all__ = ["PlanarArm", "PlanarIK", "NoCollision", "Wall"]
